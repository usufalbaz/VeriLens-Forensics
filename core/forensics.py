
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Union

import cv2
import numpy as np
from PIL import Image, ImageChops, ImageOps


ImageInput = Union[str, os.PathLike, np.ndarray, Image.Image]


class ForensicEngine:
    """
    Error Level Analysis and spatial image diagnostics.

    ELA highlights differences introduced by JPEG recompression.
    These differences are indicators only and do not prove tampering.
    """

    def __init__(
        self,
        quality: int = 90,
        scale_factor: float = 15.0,
        block_grid: int = 8,
        anomaly_percentile: float = 95.0,
        local_window: int = 15,
    ) -> None:

        if not isinstance(quality, int) or not 1 <= quality <= 100:
            raise ValueError("quality must be an integer between 1 and 100.")

        if scale_factor <= 0:
            raise ValueError("scale_factor must be greater than zero.")

        if not isinstance(block_grid, int) or not 1 <= block_grid <= 64:
            raise ValueError("block_grid must be between 1 and 64.")

        if not 0 <= anomaly_percentile <= 100:
            raise ValueError("anomaly_percentile must be between 0 and 100.")

        if not isinstance(local_window, int) or local_window < 3:
            raise ValueError("local_window must be an integer >= 3.")

        self.quality = quality
        self.scale_factor = float(scale_factor)
        self.block_grid = block_grid
        self.anomaly_percentile = float(anomaly_percentile)
        self.local_window = local_window

    @staticmethod
    def _load_image(image_input: ImageInput) -> Image.Image:
        """Load supported image inputs and return an independent RGB image."""

        if isinstance(image_input, (str, os.PathLike)):
            path = os.fspath(image_input)

            if not os.path.isfile(path):
                raise FileNotFoundError(f"Image not found: {path}")

            with Image.open(path) as source:
                corrected = ImageOps.exif_transpose(source)
                return corrected.convert("RGB").copy()

        if isinstance(image_input, Image.Image):
            corrected = ImageOps.exif_transpose(image_input)
            return corrected.convert("RGB").copy()

        if isinstance(image_input, np.ndarray):
            array = np.asarray(image_input)

            if array.size == 0:
                raise ValueError("Image array is empty.")

            if not np.issubdtype(array.dtype, np.number):
                raise TypeError("Image array must contain numeric values.")

            if not np.isfinite(array).all():
                raise ValueError("Image array contains non-finite values.")

            if array.ndim == 2:
                pass

            elif array.ndim == 3 and array.shape[2] in (1, 3, 4):
                if array.shape[2] == 1:
                    array = array[:, :, 0]

            else:
                raise ValueError("Unsupported image array shape.")

            if np.issubdtype(array.dtype, np.floating):
                if array.min() >= 0.0 and array.max() <= 1.0:
                    array = array * 255.0

            elif array.dtype == np.uint16:
                array = array.astype(np.float32) * (255.0 / 65535.0)

            array = np.clip(array, 0, 255).astype(np.uint8)

            if array.ndim == 2:
                return Image.fromarray(array).convert("RGB")

            if array.shape[2] == 3:
                return Image.fromarray(array).convert("RGB")

            if array.shape[2] == 4:
                return Image.fromarray(array, mode="RGBA").convert("RGB")

        raise TypeError("Unsupported image input type.")

    @staticmethod
    def _statistics(values: np.ndarray) -> Dict[str, float]:
        """Calculate descriptive statistics for a numeric array."""

        values = np.asarray(values, dtype=np.float32)

        if values.size == 0:
            return {
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
                "median": 0.0,
            }

        return {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "median": float(np.median(values)),
        }

    def _analyze_blocks(
        self,
        difference: np.ndarray,
    ) -> Dict[str, Any]:
        """Analyze ELA intensity across spatial image blocks."""

        height, width = difference.shape[:2]

        grid = self.block_grid
        y_edges = np.linspace(0, height, grid + 1, dtype=int)
        x_edges = np.linspace(0, width, grid + 1, dtype=int)

        blocks = []
        block_means = []

        for row in range(grid):
            for column in range(grid):
                y1 = int(y_edges[row])
                y2 = int(y_edges[row + 1])
                x1 = int(x_edges[column])
                x2 = int(x_edges[column + 1])

                region = difference[y1:y2, x1:x2]

                if region.size == 0:
                    mean_error = 0.0
                    std_error = 0.0
                else:
                    mean_error = float(np.mean(region))
                    std_error = float(np.std(region))

                block_means.append(mean_error)

                blocks.append({
                    "row": row,
                    "column": column,
                    "bounding_box": [x1, y1, x2 - x1, y2 - y1],
                    "mean_error": mean_error,
                    "std_error": std_error,
                })

        if block_means:
            highest_index = int(np.argmax(block_means))
            highest_block = blocks[highest_index]
        else:
            highest_index = None
            highest_block = None

        return {
            "grid_size": [grid, grid],
            "blocks": blocks,
            "block_mean_statistics": self._statistics(
                np.asarray(block_means, dtype=np.float32)
            ),
            "highest_mean_error_block": highest_block,
            "highest_mean_error_block_index": highest_index,
        }

    def _analyze_local_statistics(
        self,
        difference: np.ndarray,
    ) -> Dict[str, Any]:
        """Calculate local mean and standard deviation maps."""

        gray = cv2.cvtColor(difference, cv2.COLOR_RGB2GRAY)

        window = self.local_window

        if window % 2 == 0:
            window += 1

        gray_float = gray.astype(np.float32)

        local_mean = cv2.boxFilter(
            gray_float,
            ddepth=-1,
            ksize=(window, window),
            normalize=True,
            borderType=cv2.BORDER_REFLECT,
        )

        local_mean_square = cv2.boxFilter(
            gray_float * gray_float,
            ddepth=-1,
            ksize=(window, window),
            normalize=True,
            borderType=cv2.BORDER_REFLECT,
        )

        variance = np.maximum(
            local_mean_square - local_mean * local_mean,
            0.0,
        )

        local_std = np.sqrt(variance)

        return {
            "local_mean_map": local_mean,
            "local_std_map": local_std,
            "window_size": window,
            "mean_statistics": self._statistics(local_mean),
            "std_statistics": self._statistics(local_std),
        }

    def _build_anomaly_mask(
        self,
        difference: np.ndarray,
    ) -> Dict[str, Any]:
        """Create a percentile-based exploratory anomaly mask."""

        gray = cv2.cvtColor(difference, cv2.COLOR_RGB2GRAY)

        threshold = float(
            np.percentile(gray, self.anomaly_percentile)
        )

        mask = (gray > threshold).astype(np.uint8) * 255

        total_pixels = int(mask.size)
        anomaly_pixels = int(np.count_nonzero(mask))

        anomaly_ratio = (
            anomaly_pixels / total_pixels
            if total_pixels > 0
            else 0.0
        )

        return {
            "anomaly_mask": mask,
            "threshold": threshold,
            "percentile": self.anomaly_percentile,
            "anomaly_pixels": anomaly_pixels,
            "total_pixels": total_pixels,
            "anomaly_pixel_ratio": float(anomaly_ratio),
        }

    @staticmethod
    def _create_heatmap(
        difference: np.ndarray,
        scale_factor: float,
    ) -> np.ndarray:
        """Generate an RGB visualization of amplified ELA differences."""

        amplified = np.clip(
            difference.astype(np.float32) * scale_factor,
            0,
            255,
        ).astype(np.uint8)

        gray = cv2.cvtColor(amplified, cv2.COLOR_RGB2GRAY)

        colored = cv2.applyColorMap(
            gray,
            cv2.COLORMAP_JET,
        )

        return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)

    def analyze_ela(
        self,
        image_input: ImageInput,
    ) -> Dict[str, Any]:
        """
        Perform JPEG Error Level Analysis.

        Returned scores are exploratory measurements, not probabilities
        of manipulation or calibrated authenticity scores.
        """

        original = self._load_image(image_input)

        original_array = np.asarray(
            original,
            dtype=np.uint8,
        )

        height, width = original_array.shape[:2]

        if height < 1 or width < 1:
            raise ValueError("Image dimensions are invalid.")

        from io import BytesIO

        buffer = BytesIO()

        original.save(
            buffer,
            format="JPEG",
            quality=self.quality,
            optimize=False,
        )

        buffer.seek(0)

        with Image.open(buffer) as compressed_image:
            compressed = compressed_image.convert("RGB").copy()

        difference_image = ImageChops.difference(
            original,
            compressed,
        )

        difference = np.asarray(
            difference_image,
            dtype=np.uint8,
        )

        difference_float = difference.astype(np.float32)

        mean_error = float(np.mean(difference_float))
        std_error = float(np.std(difference_float))
        max_error = float(np.max(difference_float))

        # This is an uncalibrated descriptive score, not a probability.
        anomaly_score = float(
            np.clip((std_error / 50.0) * 100.0, 0.0, 100.0)
        )

        channel_statistics = {}

        channel_names = ("red", "green", "blue")

        for index, channel_name in enumerate(channel_names):
            channel_statistics[channel_name] = self._statistics(
                difference_float[:, :, index]
            )

        overall_statistics = self._statistics(
            difference_float
        )

        block_analysis = self._analyze_blocks(
            difference_float
        )

        local_analysis = self._analyze_local_statistics(
            difference
        )

        threshold_analysis = self._build_anomaly_mask(
            difference
        )

        heatmap = self._create_heatmap(
            difference,
            self.scale_factor,
        )

        return {
            # Legacy-compatible fields
            "mean_error": mean_error,
            "std_error": std_error,
            "max_error": max_error,
            "anomaly_score": anomaly_score,
            "tampering_mask": heatmap,

            # Explicit score interpretation
            "score_type": "uncalibrated_heuristic",
            "verdict": "INCONCLUSIVE",

            # ELA data
            "raw_difference_map": difference,
            "heatmap": heatmap,

            # Statistical analysis
            "overall_statistics": overall_statistics,
            "channel_statistics": channel_statistics,

            # Spatial analysis
            "block_analysis": block_analysis,
            "local_statistics": {
                "window_size": local_analysis["window_size"],
                "mean_statistics": local_analysis["mean_statistics"],
                "std_statistics": local_analysis["std_statistics"],
            },
            "local_mean_map": local_analysis["local_mean_map"],
            "local_std_map": local_analysis["local_std_map"],

            # Threshold analysis
            "anomaly_mask": threshold_analysis["anomaly_mask"],
            "threshold_analysis": {
                "threshold": threshold_analysis["threshold"],
                "percentile": threshold_analysis["percentile"],
                "anomaly_pixels": threshold_analysis["anomaly_pixels"],
                "total_pixels": threshold_analysis["total_pixels"],
                "anomaly_pixel_ratio": threshold_analysis[
                    "anomaly_pixel_ratio"
                ],
            },

            # Image information
            "image_metadata": {
                "width": int(width),
                "height": int(height),
                "channels": 3,
                "format": original.format or "unknown",
                "mode": original.mode,
            },

            # Configuration
            "configuration": {
                "jpeg_quality": self.quality,
                "scale_factor": self.scale_factor,
                "block_grid": self.block_grid,
                "anomaly_percentile": self.anomaly_percentile,
                "local_window": self.local_window,
            },

            "analysis_method": "jpeg_error_level_analysis",

            "limitations": [
                "ELA does not independently prove image manipulation.",
                "JPEG recompression can create differences in authentic images.",
                "Different image histories can produce different ELA patterns.",
                "Resizing, editing, and compression affect the measurements.",
                "The anomaly score is not a calibrated probability.",
                "The anomaly mask is a threshold visualization, not a verified tampering mask.",
                "Authenticity requires multiple independent forensic signals.",
            ],
        }

    def analyze(
        self,
        image_input: ImageInput,
    ) -> Dict[str, Any]:
        """Compatibility alias for the ELA analysis method."""

        return self.analyze_ela(image_input)
