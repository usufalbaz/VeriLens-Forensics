
"""
VeriLens Forensics - Advanced Spatial & Compression Analysis Engine.

Provides descriptive image-forensics measurements using:
- JPEG recompression difference (ELA).
- Spatial anomaly mapping.
- RGB channel difference statistics.
- Localized block analysis.
- Difference distribution statistics.
- Adaptive visualization thresholds.

All outputs are heuristic indicators.
They are not proof of manipulation or authenticity.
"""

import io
import os
from typing import Any, Dict, Union

import cv2
import numpy as np
from PIL import Image, ImageChops, ImageOps


ImageInput = Union[str, os.PathLike, np.ndarray, Image.Image]


class ForensicEngine:
    """Advanced spatial and compression analysis engine."""

    def __init__(
        self,
        quality: int = 90,
        scale_factor: float = 15.0,
        block_grid: int = 8,
        anomaly_percentile: float = 95.0,
        local_window: int = 15,
    ) -> None:

        if isinstance(quality, bool) or not isinstance(quality, int):
            raise TypeError("quality must be an integer.")

        if not 1 <= quality <= 100:
            raise ValueError("quality must be between 1 and 100.")

        if isinstance(scale_factor, bool) or not isinstance(
            scale_factor, (int, float)
        ):
            raise TypeError("scale_factor must be numeric.")

        if not np.isfinite(scale_factor) or scale_factor <= 0:
            raise ValueError("scale_factor must be finite and positive.")

        if (
            isinstance(block_grid, bool)
            or not isinstance(block_grid, int)
            or block_grid < 1
            or block_grid > 64
        ):
            raise ValueError("block_grid must be an integer from 1 to 64.")

        if isinstance(anomaly_percentile, bool) or not isinstance(
            anomaly_percentile, (int, float)
        ):
            raise TypeError("anomaly_percentile must be numeric.")

        if (
            not np.isfinite(anomaly_percentile)
            or not 50.0 <= anomaly_percentile <= 100.0
        ):
            raise ValueError(
                "anomaly_percentile must be between 50 and 100."
            )

        if (
            isinstance(local_window, bool)
            or not isinstance(local_window, int)
            or local_window < 3
            or local_window > 255
            or local_window % 2 == 0
        ):
            raise ValueError(
                "local_window must be an odd integer from 3 to 255."
            )

        self.quality = quality
        self.scale_factor = float(scale_factor)
        self.block_grid = block_grid
        self.anomaly_percentile = float(anomaly_percentile)
        self.local_window = local_window

    @staticmethod
    def _to_uint8(array: np.ndarray) -> np.ndarray:
        """Validate and convert grayscale or color arrays to uint8."""

        if not isinstance(array, np.ndarray):
            raise TypeError("Expected a NumPy array.")

        if array.size == 0:
            raise ValueError("Image array must not be empty.")

        if not (
            np.issubdtype(array.dtype, np.integer)
            or np.issubdtype(array.dtype, np.floating)
        ):
            raise TypeError("Image array must contain numeric values.")

        if not np.all(np.isfinite(array)):
            raise ValueError("Image array contains non-finite values.")

        minimum = float(np.min(array))
        maximum = float(np.max(array))

        if minimum < 0:
            raise ValueError("Negative image values are not supported.")

        if np.issubdtype(array.dtype, np.integer):
            if maximum <= 255:
                return array.astype(np.uint8)

            if array.dtype.itemsize == 2 and maximum <= 65535:
                scaled = array.astype(np.float32) / 65535.0
                return np.clip(
                    np.rint(scaled * 255.0), 0, 255
                ).astype(np.uint8)

            raise ValueError("Unsupported integer image range.")

        if maximum <= 1.0:
            array = array.astype(np.float32) * 255.0
        elif maximum <= 255.0:
            array = array.astype(np.float32)
        else:
            raise ValueError(
                "Floating-point images must be in [0, 1] or [0, 255]."
            )

        return np.clip(np.rint(array), 0, 255).astype(np.uint8)

    @staticmethod
    def _load_image(image_input: ImageInput) -> Image.Image:
        """Convert supported inputs into an independent RGB image."""

        if isinstance(image_input, (str, os.PathLike)):
            path = os.fspath(image_input)

            if not os.path.isfile(path):
                raise FileNotFoundError(f"Image not found: {path}")

            with Image.open(path) as source:
                image = ImageOps.exif_transpose(source)
                return image.convert("RGB").copy()

        if isinstance(image_input, Image.Image):
            image = ImageOps.exif_transpose(image)
            return image.convert("RGB").copy()

        if isinstance(image_input, np.ndarray):
            if image_input.ndim not in (2, 3):
                raise ValueError(
                    "Expected a grayscale or color image array."
                )

            if image_input.ndim == 3 and image_input.shape[2] not in (
                1, 3, 4
            ):
                raise ValueError(
                    "Expected 1, 3, or 4 image channels."
                )

            array = ForensicEngine._to_uint8(image_input)

            if array.ndim == 3 and array.shape[2] == 1:
                array = array[:, :, 0]

            if array.ndim == 2:
                return Image.fromarray(array).convert("RGB")

            if array.shape[2] == 4:
                rgba = Image.fromarray(array, mode="RGBA")
                background = Image.new(
                    "RGBA",
                    rgba.size,
                    (255, 255, 255, 255),
                )
                return Image.alpha_composite(
                    background,
                    rgba,
                ).convert("RGB")

            # NumPy color arrays are interpreted as RGB.
            return Image.fromarray(array, mode="RGB").copy()

        raise TypeError(
            "Expected a file path, NumPy array, or PIL Image instance."
        )

    @staticmethod
    def _calculate_statistics(
        values: np.ndarray,
    ) -> Dict[str, float]:
        """Calculate descriptive statistics for numeric values."""

        data = np.asarray(values, dtype=np.float64)

        if data.size == 0:
            return {
                "mean": 0.0,
                "std": 0.0,
                "median": 0.0,
                "minimum": 0.0,
                "maximum": 0.0,
                "p90": 0.0,
                "p95": 0.0,
                "p99": 0.0,
            }

        return {
            "mean": float(np.mean(data)),
            "std": float(np.std(data)),
            "median": float(np.median(data)),
            "minimum": float(np.min(data)),
            "maximum": float(np.max(data)),
            "p90": float(np.percentile(data, 90)),
            "p95": float(np.percentile(data, 95)),
            "p99": float(np.percentile(data, 99)),
        }

    def _analyze_channels(
        self,
        difference: np.ndarray,
    ) -> Dict[str, Any]:
        """Analyze JPEG difference statistics independently per RGB channel."""

        names = ("red", "green", "blue")
        results = {}

        for index, name in enumerate(names):
            results[name] = self._calculate_statistics(
                difference[:, :, index]
            )

        return results

    def _analyze_blocks(
        self,
        difference_gray: np.ndarray,
    ) -> Dict[str, Any]:
        """Measure spatial difference statistics in image blocks."""

        height, width = difference_gray.shape
        grid = self.block_grid

        row_edges = np.linspace(
            0, height, grid + 1, dtype=np.int32
        )
        col_edges = np.linspace(
            0, width, grid + 1, dtype=np.int32
        )

        block_means = []
        block_stds = []
        block_maxima = []

        block_details = []

        for row in range(grid):
            for col in range(grid):
                y1, y2 = row_edges[row], row_edges[row + 1]
                x1, x2 = col_edges[col], col_edges[col + 1]

                block = difference_gray[y1:y2, x1:x2]

                if block.size == 0:
                    mean_value = 0.0
                    std_value = 0.0
                    max_value = 0.0
                else:
                    mean_value = float(np.mean(block))
                    std_value = float(np.std(block))
                    max_value = float(np.max(block))

                block_means.append(mean_value)
                block_stds.append(std_value)
                block_maxima.append(max_value)

                block_details.append({
                    "row": row,
                    "column": col,
                    "x_start": int(x1),
                    "y_start": int(y1),
                    "x_end": int(x2),
                    "y_end": int(y2),
                    "mean_error": mean_value,
                    "std_error": std_value,
                    "max_error": max_value,
                })

        return {
            "grid_size": grid,
            "block_count": len(block_details),
            "mean_error_grid": np.asarray(
                block_means,
                dtype=np.float32,
            ).reshape(grid, grid).tolist(),
            "std_error_grid": np.asarray(
                block_stds,
                dtype=np.float32,
            ).reshape(grid, grid).tolist(),
            "max_error_grid": np.asarray(
                block_maxima,
                dtype=np.float32,
            ).reshape(grid, grid).tolist(),
            "blocks": block_details,
            "highest_mean_error_block": (
                int(np.argmax(block_means))
                if block_means else None
            ),
        }

    def _create_spatial_maps(
        self,
        difference_gray: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Create raw, locally averaged, and visualization maps."""

        raw_map = difference_gray.astype(np.float32)

        local_mean = cv2.boxFilter(
            raw_map,
            ddepth=-1,
            ksize=(self.local_window, self.local_window),
            normalize=True,
            borderType=cv2.BORDER_REFLECT,
        )

        local_squared_mean = cv2.boxFilter(
            np.square(raw_map),
            ddepth=-1,
            ksize=(self.local_window, self.local_window),
            normalize=True,
            borderType=cv2.BORDER_REFLECT,
        )

        local_variance = np.maximum(
            local_squared_mean - np.square(local_mean),
            0.0,
        )

        local_std = np.sqrt(local_variance)

        # This is a visualization map, not a calibrated anomaly score.
        visualization_map = np.clip(
            local_mean * self.scale_factor,
            0,
            255,
        ).astype(np.uint8)

        heatmap_bgr = cv2.applyColorMap(
            visualization_map,
            cv2.COLORMAP_JET,
        )

        heatmap_rgb = cv2.cvtColor(
            heatmap_bgr,
            cv2.COLOR_BGR2RGB,
        )

        return {
            "raw_difference_map": difference_gray,
            "local_mean_map": local_mean,
            "local_std_map": local_std,
            "visualization_map": visualization_map,
            "heatmap": heatmap_rgb,
        }

    def analyze_ela(
        self,
        image_input: ImageInput,
    ) -> Dict[str, Any]:
        """
        Perform ELA and extended spatial compression analysis.

        Returns the historical ELA fields and additional diagnostics.
        """

        original = self._load_image(image_input)

        width, height = original.size

        if width < 2 or height < 2:
            raise ValueError("Image dimensions must be at least 2x2.")

        buffer = io.BytesIO()

        try:
            original.save(
                buffer,
                format="JPEG",
                quality=self.quality,
                subsampling=0,
                optimize=False,
            )

            buffer.seek(0)

            with Image.open(buffer) as recompressed_image:
                recompressed = recompressed_image.convert(
                    "RGB"
                ).copy()

        finally:
            buffer.close()

        if original.size != recompressed.size:
            raise RuntimeError(
                "Recompressed image dimensions changed unexpectedly."
            )

        difference = ImageChops.difference(
            original,
            recompressed,
        )

        difference_array = np.asarray(
            difference,
            dtype=np.uint8,
        )

        difference_gray = cv2.cvtColor(
            difference_array,
            cv2.COLOR_RGB2GRAY,
        )

        # All core measurements use unamplified difference data.
        overall_statistics = self._calculate_statistics(
            difference_gray
        )

        mean_error = overall_statistics["mean"]
        std_error = overall_statistics["std"]
        max_error = overall_statistics["maximum"]

        # Retain the legacy score for API compatibility.
        anomaly_score = float(
            np.clip((std_error / 50.0) * 100.0, 0.0, 100.0)
        )

        spatial_maps = self._create_spatial_maps(
            difference_gray
        )

        threshold = float(
            np.percentile(
                difference_gray,
                self.anomaly_percentile,
            )
        )

        anomaly_mask = (
            difference_gray > threshold
        ).astype(np.uint8) * 255

        anomalous_pixel_ratio = float(
            np.mean(anomaly_mask > 0)
        )

        channel_statistics = self._analyze_channels(
            difference_array
        )

        block_analysis = self._analyze_blocks(
            difference_gray
        )

        local_statistics = {
            "local_mean": self._calculate_statistics(
                spatial_maps["local_mean_map"]
            ),
            "local_std": self._calculate_statistics(
                spatial_maps["local_std_map"]
            ),
        }

        # Preserve the legacy key "tampering_mask".
        # It remains an RGB heatmap for compatibility.
        return {
            "mean_error": mean_error,
            "std_error": std_error,
            "max_error": max_error,
            "anomaly_score": anomaly_score,
            "tampering_mask": spatial_maps["heatmap"],

            "raw_difference_map": spatial_maps[
                "raw_difference_map"
            ],
            "heatmap": spatial_maps["heatmap"],
            "anomaly_mask": anomaly_mask,
            "local_mean_map": spatial_maps["local_mean_map"],
            "local_std_map": spatial_maps["local_std_map"],

            "overall_statistics": overall_statistics,
            "channel_statistics": channel_statistics,
            "block_analysis": block_analysis,
            "local_statistics": local_statistics,

            "threshold_analysis": {
                "percentile": self.anomaly_percentile,
                "threshold_value": threshold,
                "anomalous_pixel_ratio": anomalous_pixel_ratio,
                "threshold_type": "image_relative_percentile",
            },

            "image_metadata": {
                "width": width,
                "height": height,
                "mode": "RGB",
                "pixel_count": width * height,
            },

            "configuration": {
                "jpeg_quality": self.quality,
                "scale_factor": self.scale_factor,
                "block_grid": self.block_grid,
                "anomaly_percentile": self.anomaly_percentile,
                "local_window": self.local_window,
            },

            "analysis_method": (
                "JPEG recompression difference with spatial diagnostics"
            ),
            "score_type": "Uncalibrated heuristic anomaly score",

            "limitations": [
                "ELA differences do not prove image manipulation.",
                "JPEG quality and compression history affect the result.",
                "Image content and texture can create non-uniform differences.",
                "The percentile mask is relative to the analyzed image.",
                "Block statistics do not identify the cause of an anomaly.",
                "Channel differences may reflect normal color and compression behavior.",
                "The heatmap visualizes compression differences, not confirmed edits.",
                "This method is not a standalone AI-image detector.",
                "The anomaly score is not a probability.",
            ],
        }

    def analyze(
        self,
        image_input: ImageInput,
    ) -> Dict[str, Any]:
        """Run the complete spatial and compression analysis."""

        return self.analyze_ela(image_input)
