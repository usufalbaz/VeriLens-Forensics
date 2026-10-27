
"""
VeriLens Forensics - Spatial & Compression Analysis Engine

Uses Error Level Analysis (ELA) to visualize differences between an image
and a JPEG-recompressed copy. ELA findings are heuristic indicators only.
"""

import io
import os
from typing import Any, Dict, Union

import cv2
import numpy as np
from PIL import Image, ImageChops


ImageInput = Union[str, os.PathLike, np.ndarray, Image.Image]


class ForensicEngine:
    """Compute ELA difference statistics and a visualization heatmap."""

    def __init__(self, quality: int = 90, scale_factor: float = 15.0):
        if isinstance(quality, bool) or not isinstance(quality, int):
            raise TypeError("quality must be an integer from 1 to 100.")
        if not 1 <= quality <= 100:
            raise ValueError("quality must be between 1 and 100.")

        if isinstance(scale_factor, bool) or not isinstance(
            scale_factor, (int, float)
        ):
            raise TypeError("scale_factor must be a positive number.")
        if not np.isfinite(scale_factor) or scale_factor <= 0:
            raise ValueError("scale_factor must be finite and positive.")

        self.quality = quality
        self.scale_factor = float(scale_factor)

    @staticmethod
    def _load_image(image_input: ImageInput) -> Image.Image:
        """Convert supported inputs into an independent RGB image."""

        if isinstance(image_input, (str, os.PathLike)):
            path = os.fspath(image_input)
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Image not found: {path}")

            with Image.open(path) as source:
                return source.convert("RGB").copy()

        if isinstance(image_input, Image.Image):
            return image_input.convert("RGB").copy()

        if isinstance(image_input, np.ndarray):
            if image_input.size == 0:
                raise ValueError("Image array must not be empty.")

            if image_input.ndim not in (2, 3):
                raise ValueError("Expected a grayscale or color image array.")

            if image_input.ndim == 3 and image_input.shape[2] not in (1, 3, 4):
                raise ValueError("Expected 1, 3, or 4 image channels.")

            if image_input.dtype != np.uint8:
                if not np.issubdtype(image_input.dtype, np.number):
                    raise TypeError("Image array must contain numeric values.")
                if not np.all(np.isfinite(image_input)):
                    raise ValueError("Image array contains non-finite values.")

                image_input = np.clip(image_input, 0, 255).astype(np.uint8)

            if image_input.ndim == 3 and image_input.shape[2] == 1:
                image_input = image_input[:, :, 0]

            return Image.fromarray(image_input).convert("RGB")

        raise TypeError(
            "Expected a file path, NumPy array, or PIL Image instance."
        )

    def analyze_ela(self, image_input: ImageInput) -> Dict[str, Any]:
        """
        Compare the image with a JPEG-recompressed copy.

        Returned scores are heuristic measurements, not probabilities
        of forgery or proof that manipulation occurred.
        """

        original = self._load_image(image_input)

        buffer = io.BytesIO()
        try:
            original.save(
                buffer,
                format="JPEG",
                quality=self.quality,
            )
            buffer.seek(0)

            with Image.open(buffer) as recompressed_image:
                recompressed = recompressed_image.convert("RGB").copy()
        finally:
            buffer.close()

        difference = ImageChops.difference(original, recompressed)
        difference_array = np.asarray(difference, dtype=np.uint8)

        # Statistics are measured on the unamplified difference.
        mean_error = float(np.mean(difference_array))
        std_error = float(np.std(difference_array))
        max_error = float(np.max(difference_array))

        # This is an uncalibrated heuristic, not a forgery probability.
        anomaly_score = float(
            np.clip((std_error / 50.0) * 100.0, 0.0, 100.0)
        )

        # Amplification is used only to improve visualization.
        amplified = np.clip(
            difference_array.astype(np.float32) * self.scale_factor,
            0,
            255,
        ).astype(np.uint8)

        gray_difference = cv2.cvtColor(
            amplified,
            cv2.COLOR_RGB2GRAY,
        )

        heatmap_bgr = cv2.applyColorMap(
            gray_difference,
            cv2.COLORMAP_JET,
        )
        heatmap_rgb = cv2.cvtColor(
            heatmap_bgr,
            cv2.COLOR_BGR2RGB,
        )

        return {
            "mean_error": mean_error,
            "std_error": std_error,
            "max_error": max_error,
            "anomaly_score": anomaly_score,
            "tampering_mask": heatmap_rgb,
            "analysis_method": "JPEG recompression difference (ELA)",
            "score_type": "Uncalibrated heuristic anomaly score",
            "limitations": [
                "ELA differences do not prove image manipulation.",
                "JPEG quality and compression history affect the result.",
                "Uniform or absent differences do not prove authenticity.",
                "This method is not a reliable standalone detector for AI images.",
                "The heatmap highlights compression differences, not confirmed edits.",
            ],
        }
