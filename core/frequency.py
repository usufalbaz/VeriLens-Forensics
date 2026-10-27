
"""
VeriLens Forensics - Frequency Domain Analysis Engine.

Computes descriptive Fourier-domain features for digital image inspection.

Important:
- These features describe an image's frequency distribution.
- They are not, by themselves, evidence of manipulation or AI generation.
- No authenticity probability is produced by this module.
"""

from pathlib import Path
from typing import Any, Dict, Tuple, Union

import cv2
import numpy as np
from PIL import Image


ImageInput = Union[str, Path, np.ndarray, Image.Image]


class FrequencyEngine:
    """Extract descriptive frequency-domain features from an image."""

    def __init__(
        self,
        target_size: Tuple[int, int] = (256, 256),
        num_bands: int = 64,
        high_frequency_start: float = 0.60,
    ) -> None:
        """
        Args:
            target_size: Output size as (width, height).
            num_bands: Number of radial frequency bands.
            high_frequency_start: Fraction of the inscribed radius at which
                the high-frequency region begins. Must be in [0, 1).

        Raises:
            ValueError: If configuration values are invalid.
        """
        if (
            not isinstance(target_size, (tuple, list))
            or len(target_size) != 2
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, np.integer))
                or value <= 0
                for value in target_size
            )
        ):
            raise ValueError(
                "target_size must contain two positive integers: (width, height)."
            )

        if (
            isinstance(num_bands, bool)
            or not isinstance(num_bands, (int, np.integer))
            or num_bands < 1
        ):
            raise ValueError("num_bands must be a positive integer.")

        if (
            isinstance(high_frequency_start, bool)
            or not isinstance(high_frequency_start, (int, float))
            or not np.isfinite(high_frequency_start)
            or not 0.0 <= high_frequency_start < 1.0
        ):
            raise ValueError("high_frequency_start must be in [0, 1).")

        self.target_size = (int(target_size[0]), int(target_size[1]))
        self.num_bands = int(num_bands)
        self.high_frequency_start = float(high_frequency_start)

    @staticmethod
    def _load_image(image_input: ImageInput) -> np.ndarray:
        """
        Convert supported input types to an RGB uint8 array.

        NumPy color images are interpreted as RGB/RGBA. Grayscale arrays
        are accepted directly. Floating-point arrays must contain finite
        values in [0, 1] or [0, 255].
        """
        if isinstance(image_input, (str, Path)):
            image = cv2.imread(str(image_input), cv2.IMREAD_UNCHANGED)

            if image is None:
                raise ValueError(
                    f"Unable to read image from path: {image_input}"
                )

            if image.ndim == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.ndim == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            elif image.ndim == 3 and image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
                image = FrequencyEngine._composite_alpha(image)
            else:
                raise ValueError("Unsupported image channel layout.")

        elif isinstance(image_input, Image.Image):
            image = np.asarray(image_input.convert("RGB"))

        elif isinstance(image_input, np.ndarray):
            image = image_input.copy()

            if image.size == 0:
                raise ValueError("Image array must not be empty.")

            if image.ndim == 2:
                image = cv2.cvtColor(
                    FrequencyEngine._to_uint8(image),
                    cv2.COLOR_GRAY2RGB,
                )
            elif image.ndim == 3 and image.shape[2] == 3:
                image = FrequencyEngine._to_uint8(image)
            elif image.ndim == 3 and image.shape[2] == 4:
                image = FrequencyEngine._to_uint8(image)
                image = FrequencyEngine._composite_alpha(image)
            else:
                raise ValueError(
                    "NumPy input must be grayscale, RGB, or RGBA."
                )
        else:
            raise TypeError(
                "image_input must be a path, NumPy array, or PIL Image."
            )

        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Image conversion did not produce RGB data.")

        if image.shape[0] < 1 or image.shape[1] < 1:
            raise ValueError("Image dimensions must be positive.")

        return np.ascontiguousarray(image, dtype=np.uint8)

    @staticmethod
    def _to_uint8(array: np.ndarray) -> np.ndarray:
        """Validate and convert a numeric array to uint8."""
        if not np.issubdtype(array.dtype, np.number):
            raise TypeError("Image arrays must have a numeric data type.")

        if not np.all(np.isfinite(array)):
            raise ValueError("Image arrays must contain only finite values.")

        minimum = float(np.min(array))
        maximum = float(np.max(array))

        if minimum < 0:
            raise ValueError("Image values must not be negative.")

        if maximum <= 1.0:
            array = array.astype(np.float32) * 255.0
        elif maximum <= 255.0:
            array = array.astype(np.float32)
        else:
            raise ValueError(
                "Image values must be in [0, 1] or [0, 255]."
            )

        return np.clip(np.rint(array), 0, 255).astype(np.uint8)

    @staticmethod
    def _composite_alpha(rgba: np.ndarray) -> np.ndarray:
        """
        Composite RGBA pixels over white.

        Alpha handling is explicit so transparent pixels do not create
        arbitrary black regions in frequency analysis.
        """
        rgba_float = rgba.astype(np.float32)
        rgb = rgba_float[..., :3]
        alpha = rgba_float[..., 3:4] / 255.0

        composited = rgb * alpha + 255.0 * (1.0 - alpha)

        return np.clip(np.rint(composited), 0, 255).astype(np.uint8)

    def extract_features(self, image_input: ImageInput) -> Dict[str, Any]:
        """
        Extract radial spectrum statistics and a display heatmap.

        Returns:
            feature_vector:
                Radial log-magnitude means, radial log-magnitude standard
                deviations, and the high-frequency power ratio.
            hf_energy_ratio:
                Fraction of non-DC spectral power inside the selected
                high-frequency annulus. This is descriptive, not a
                manipulation probability.
            mean_variance:
                Mean radial standard deviation of log magnitude. The
                historical key is retained for API compatibility.
            heatmap:
                RGB visualization of log spectral magnitude.
            radial_mean:
                Mean log magnitude for each radial band.
            radial_std:
                Standard deviation of log magnitude for each radial band.
            spectral_metrics:
                Explicitly named descriptive metrics and their definitions.
        """
        image = self._load_image(image_input)

        width, height = self.target_size
        resized = cv2.resize(
            image,
            (width, height),
            interpolation=cv2.INTER_AREA
            if image.shape[1] > width or image.shape[0] > height
            else cv2.INTER_LINEAR,
        )

        gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
        gray_float = gray.astype(np.float32)

        # Remove the mean to reduce the DC component's influence.
        centered = gray_float - float(np.mean(gray_float))

        # A Hann window reduces edge discontinuities introduced by resizing
        # and cropping. Its use is fixed to keep outputs reproducible.
        window_y = np.hanning(height).astype(np.float32)
        window_x = np.hanning(width).astype(np.float32)
        window = np.outer(window_y, window_x)

        windowed = centered * window

        spectrum = np.fft.fftshift(np.fft.fft2(windowed))
        magnitude = np.abs(spectrum).astype(np.float64)

        # Log magnitude is used only for robust descriptive radial statistics
        # and visualization. It is not used to calculate power ratios.
        log_magnitude = np.log1p(magnitude)

        # Power is non-negative and proportional to squared magnitude.
        power = np.square(magnitude)

        y, x = np.indices((height, width))
        cy = (height - 1) / 2.0
        cx = (width - 1) / 2.0
        radius = np.hypot(x - cx, y - cy)

        max_radius = min(width, height) / 2.0
        if max_radius <= 0:
            raise ValueError("Target image dimensions are too small.")

        radial_mean = []
        radial_std = []

        # Use evenly spaced radial bands, with a stable output length.
        band_edges = np.linspace(
            0.0,
            max_radius,
            self.num_bands + 1,
            dtype=np.float64,
        )

        for index in range(self.num_bands):
            lower = band_edges[index]
            upper = band_edges[index + 1]

            if index == self.num_bands - 1:
                mask = (radius >= lower) & (radius <= upper)
            else:
                mask = (radius >= lower) & (radius < upper)

            values = log_magnitude[mask]

            if values.size:
                radial_mean.append(float(np.mean(values)))
                radial_std.append(float(np.std(values)))
            else:
                radial_mean.append(0.0)
                radial_std.append(0.0)

        # Exclude the exact center/DC coefficient from the power ratio.
        valid_spectrum = radius <= max_radius
        dc_y = int(round(cy))
        dc_x = int(round(cx))
        valid_spectrum[dc_y, dc_x] = False

        high_frequency_mask = (
            valid_spectrum
            & (radius >= max_radius * self.high_frequency_start)
        )

        total_power = float(np.sum(power[valid_spectrum], dtype=np.float64))
        high_frequency_power = float(
            np.sum(power[high_frequency_mask], dtype=np.float64)
        )

        if total_power > np.finfo(np.float64).eps:
            hf_energy_ratio = float(
                np.clip(high_frequency_power / total_power, 0.0, 1.0)
            )
        else:
            # Constant or zero-variance images have no meaningful
            # non-DC power distribution.
            hf_energy_ratio = 0.0

        mean_radial_std = float(np.mean(radial_std))

        # Normalize only for display. Display intensities are not forensic
        # scores and must not be interpreted as probabilities.
        normalized = cv2.normalize(
            log_magnitude,
            None,
            alpha=0,
            beta=255,
            norm_type=cv2.NORM_MINMAX,
        )

        heatmap_bgr = cv2.applyColorMap(
            normalized.astype(np.uint8),
            cv2.COLORMAP_INFERNO,
        )
        heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

        feature_vector = np.asarray(
            radial_mean + radial_std + [hf_energy_ratio],
            dtype=np.float32,
        )

        return {
            "feature_vector": feature_vector,
            "hf_energy_ratio": hf_energy_ratio,
            "mean_variance": mean_radial_std,
            "heatmap": heatmap_rgb,
            "radial_mean": radial_mean,
            "radial_std": radial_std,
            "spectral_metrics": {
                "high_frequency_power_ratio": hf_energy_ratio,
                "total_non_dc_power": total_power,
                "high_frequency_power": high_frequency_power,
                "high_frequency_start": self.high_frequency_start,
                "radial_band_count": self.num_bands,
                "metric_type": "descriptive_frequency_statistic",
                "is_authenticity_probability": False,
            },
        }
