
"""
VeriLens Forensics - Advanced Frequency Domain Analysis Engine.

Provides descriptive spectral analysis for digital image inspection.

Features:
- Two-dimensional Fast Fourier Transform (2D-FFT).
- Radial frequency decomposition.
- Radial power distribution.
- High-frequency energy ratio.
- Spectral entropy.
- Spectral centroid.
- Angular energy distribution.
- Spectral anisotropy.
- Log-magnitude visualization.

All measurements are descriptive statistical indicators.
They are not calibrated authenticity or AI-generation probabilities.
"""

from pathlib import Path
from typing import Any, Dict, Tuple, Union

import cv2
import numpy as np
from PIL import Image


ImageInput = Union[str, Path, np.ndarray, Image.Image]


class FrequencyEngine:
    """Advanced descriptive frequency-domain feature extractor."""

    def __init__(
        self,
        target_size: Tuple[int, int] = (256, 256),
        num_bands: int = 64,
        high_frequency_start: float = 0.60,
        angular_bins: int = 36,
        apply_window: bool = True,
    ) -> None:

        if (
            not isinstance(target_size, (tuple, list))
            or len(target_size) != 2
            or any(
                isinstance(v, bool)
                or not isinstance(v, (int, np.integer))
                or v < 2
                for v in target_size
            )
        ):
            raise ValueError(
                "target_size must contain two integers >= 2."
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
            raise ValueError(
                "high_frequency_start must be in [0, 1)."
            )

        if (
            isinstance(angular_bins, bool)
            or not isinstance(angular_bins, (int, np.integer))
            or angular_bins < 4
        ):
            raise ValueError("angular_bins must be an integer >= 4.")

        if not isinstance(apply_window, bool):
            raise TypeError("apply_window must be a boolean.")

        self.target_size = (int(target_size[0]), int(target_size[1]))
        self.num_bands = int(num_bands)
        self.high_frequency_start = float(high_frequency_start)
        self.angular_bins = int(angular_bins)
        self.apply_window = apply_window

    @staticmethod
    def _to_uint8(array: np.ndarray) -> np.ndarray:
        """Validate and convert supported image arrays to uint8."""

        if not isinstance(array, np.ndarray):
            raise TypeError("Expected a NumPy array.")

        if array.size == 0:
            raise ValueError("Image array must not be empty.")

        if not (
            np.issubdtype(array.dtype, np.integer)
            or np.issubdtype(array.dtype, np.floating)
        ):
            raise TypeError("Image arrays must contain numeric values.")

        if not np.all(np.isfinite(array)):
            raise ValueError("Image arrays must contain finite values.")

        minimum = float(np.min(array))
        maximum = float(np.max(array))

        if minimum < 0:
            raise ValueError("Negative image values are not supported.")

        if np.issubdtype(array.dtype, np.integer):
            if maximum <= 255:
                return np.clip(array, 0, 255).astype(np.uint8)

            if array.dtype.itemsize == 2 and maximum <= 65535:
                scaled = array.astype(np.float32) / 65535.0
                return np.clip(
                    np.rint(scaled * 255.0), 0, 255
                ).astype(np.uint8)

            raise ValueError(
                "Integer images must use an 8-bit or 16-bit range."
            )

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
    def _composite_alpha(rgba: np.ndarray) -> np.ndarray:
        """Composite an RGBA image over a white background."""

        rgba = FrequencyEngine._to_uint8(rgba)

        rgb = rgba[..., :3].astype(np.float32)
        alpha = rgba[..., 3:4].astype(np.float32) / 255.0

        result = rgb * alpha + 255.0 * (1.0 - alpha)

        return np.clip(np.rint(result), 0, 255).astype(np.uint8)

    @staticmethod
    def _load_image(image_input: ImageInput) -> np.ndarray:
        """Load supported image inputs and return contiguous RGB uint8."""

        if isinstance(image_input, (str, Path)):
            image = cv2.imread(
                str(image_input),
                cv2.IMREAD_UNCHANGED,
            )

            if image is None:
                raise ValueError(
                    f"Unable to read image: {image_input}"
                )

            if image.ndim == 2:
                image = FrequencyEngine._to_uint8(image)
                image = cv2.cvtColor(
                    image,
                    cv2.COLOR_GRAY2RGB,
                )

            elif image.ndim == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(
                    image,
                    cv2.COLOR_BGR2RGB,
                )
                image = FrequencyEngine._to_uint8(image)

            elif image.ndim == 3 and image.shape[2] == 4:
                image = cv2.cvtColor(
                    image,
                    cv2.COLOR_BGRA2RGBA,
                )
                image = FrequencyEngine._composite_alpha(image)

            else:
                raise ValueError("Unsupported image channel layout.")

        elif isinstance(image_input, Image.Image):
            image = np.asarray(
                image_input.convert("RGBA")
            )
            image = FrequencyEngine._composite_alpha(image)

        elif isinstance(image_input, np.ndarray):
            image = image_input.copy()

            if image.ndim == 2:
                image = FrequencyEngine._to_uint8(image)
                image = cv2.cvtColor(
                    image,
                    cv2.COLOR_GRAY2RGB,
                )

            elif image.ndim == 3 and image.shape[2] == 3:
                image = FrequencyEngine._to_uint8(image)

            elif image.ndim == 3 and image.shape[2] == 4:
                image = FrequencyEngine._composite_alpha(image)

            else:
                raise ValueError(
                    "NumPy input must be grayscale, RGB, or RGBA."
                )

        else:
            raise TypeError(
                "Input must be a path, NumPy array, or PIL Image."
            )

        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Expected an RGB image.")

        if image.shape[0] < 2 or image.shape[1] < 2:
            raise ValueError("Image dimensions must be at least 2x2.")

        return np.ascontiguousarray(image, dtype=np.uint8)

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        """Resize an image using an interpolation suited to its dimensions."""

        width, height = self.target_size

        shrinking = (
            image.shape[1] > width
            or image.shape[0] > height
        )

        interpolation = (
            cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR
        )

        return cv2.resize(
            image,
            (width, height),
            interpolation=interpolation,
        )

    def _build_frequency_geometry(
        self,
        height: int,
        width: int,
    ) -> Dict[str, Any]:
        """Build centered radial and angular frequency coordinates."""

        # fftshift places the DC coefficient at integer center indices.
        center_y = height // 2
        center_x = width // 2

        y, x = np.indices((height, width))

        dx = x - center_x
        dy = y - center_y

        radius = np.hypot(dx, dy)
        angle = np.arctan2(dy, dx)

        max_radius = min(height, width) / 2.0

        valid_mask = radius <= max_radius

        # Exclude the exact DC coefficient.
        valid_mask[center_y, center_x] = False

        return {
            "radius": radius,
            "angle": angle,
            "max_radius": max_radius,
            "valid_mask": valid_mask,
            "center": (center_y, center_x),
        }

    def _radial_analysis(
        self,
        log_magnitude: np.ndarray,
        power: np.ndarray,
        radius: np.ndarray,
        valid_mask: np.ndarray,
        max_radius: float,
    ) -> Dict[str, Any]:
        """Calculate radial magnitude and power statistics."""

        band_indices = np.floor(
            radius / max_radius * self.num_bands
        ).astype(np.int32)

        band_indices = np.clip(
            band_indices,
            0,
            self.num_bands - 1,
        )

        radial_mean = []
        radial_std = []
        radial_power = []

        total_power = float(
            np.sum(power[valid_mask], dtype=np.float64)
        )

        for band in range(self.num_bands):
            mask = valid_mask & (band_indices == band)

            magnitudes = log_magnitude[mask]
            band_power = power[mask]

            if magnitudes.size:
                radial_mean.append(float(np.mean(magnitudes)))
                radial_std.append(float(np.std(magnitudes)))
            else:
                radial_mean.append(0.0)
                radial_std.append(0.0)

            if total_power > 0:
                radial_power.append(
                    float(np.sum(band_power, dtype=np.float64) / total_power)
                )
            else:
                radial_power.append(0.0)

        radial_power_array = np.asarray(
            radial_power,
            dtype=np.float64,
        )

        if np.sum(radial_power_array) > 0:
            peak_band = int(np.argmax(radial_power_array))

            band_centers = (
                np.arange(self.num_bands, dtype=np.float64) + 0.5
            ) / self.num_bands

            spectral_centroid = float(
                np.sum(radial_power_array * band_centers)
                / np.sum(radial_power_array)
            )

        else:
            peak_band = None
            spectral_centroid = 0.0

        high_frequency_mask = (
            valid_mask
            & (
                radius
                >= max_radius * self.high_frequency_start
            )
        )

        high_frequency_power = float(
            np.sum(power[high_frequency_mask], dtype=np.float64)
        )

        if total_power > np.finfo(np.float64).eps:
            hf_energy_ratio = float(
                np.clip(
                    high_frequency_power / total_power,
                    0.0,
                    1.0,
                )
            )
        else:
            hf_energy_ratio = 0.0

        return {
            "radial_mean": radial_mean,
            "radial_std": radial_std,
            "radial_power": radial_power,
            "total_power": total_power,
            "high_frequency_power": high_frequency_power,
            "hf_energy_ratio": hf_energy_ratio,
            "peak_frequency_band": peak_band,
            "spectral_centroid": spectral_centroid,
        }

    def _angular_analysis(
        self,
        power: np.ndarray,
        angle: np.ndarray,
        valid_mask: np.ndarray,
    ) -> Dict[str, Any]:
        """Measure the distribution of spectral power by orientation."""

        angles = (
            angle[valid_mask] + np.pi
        ) % (2.0 * np.pi)

        values = power[valid_mask]

        angular_power, _ = np.histogram(
            angles,
            bins=self.angular_bins,
            range=(0.0, 2.0 * np.pi),
            weights=values,
        )

        total = float(np.sum(angular_power))

        if total > np.finfo(np.float64).eps:
            normalized = angular_power.astype(np.float64) / total

            mean_power = float(np.mean(normalized))
            std_power = float(np.std(normalized))

            anisotropy = (
                std_power / mean_power
                if mean_power > 0
                else 0.0
            )

            dominant_bin = int(np.argmax(normalized))

            dominant_angle = float(
                -180.0
                + (dominant_bin + 0.5)
                * (360.0 / self.angular_bins)
            )

        else:
            normalized = np.zeros(
                self.angular_bins,
                dtype=np.float64,
            )
            anisotropy = 0.0
            dominant_bin = None
            dominant_angle = None

        return {
            "angular_power_distribution": normalized.tolist(),
            "spectral_anisotropy": float(anisotropy),
            "dominant_orientation_degrees": dominant_angle,
            "dominant_orientation_bin": dominant_bin,
            "angular_bin_count": self.angular_bins,
        }

    @staticmethod
    def _calculate_spectral_entropy(
        power: np.ndarray,
        valid_mask: np.ndarray,
    ) -> float:
        """Calculate normalized Shannon entropy of non-DC spectral power."""

        values = power[valid_mask].astype(np.float64)

        total = float(np.sum(values))

        if total <= np.finfo(np.float64).eps:
            return 0.0

        probabilities = values / total
        probabilities = probabilities[probabilities > 0]

        entropy = -float(
            np.sum(probabilities * np.log2(probabilities))
        )

        maximum_entropy = np.log2(values.size)

        if maximum_entropy <= 0:
            return 0.0

        return float(
            np.clip(entropy / maximum_entropy, 0.0, 1.0)
        )

    @staticmethod
    def _create_heatmap(log_magnitude: np.ndarray) -> np.ndarray:
        """Generate an RGB log-spectrum visualization."""

        normalized = cv2.normalize(
            log_magnitude,
            None,
            alpha=0,
            beta=255,
            norm_type=cv2.NORM_MINMAX,
        )

        normalized = np.clip(
            normalized,
            0,
            255,
        ).astype(np.uint8)

        heatmap = cv2.applyColorMap(
            normalized,
            cv2.COLORMAP_INFERNO,
        )

        return cv2.cvtColor(
            heatmap,
            cv2.COLOR_BGR2RGB,
        )

    def extract_features(
        self,
        image_input: ImageInput,
    ) -> Dict[str, Any]:
        """
        Extract descriptive spectral features.

        Historical output keys are preserved for pipeline compatibility.

        Returns:
            feature_vector:
                Radial means, radial standard deviations, and HF ratio.
            hf_energy_ratio:
                High-frequency power divided by non-DC power.
            mean_variance:
                Historical alias for mean radial standard deviation.
            heatmap:
                RGB spectral visualization.
            radial_mean:
                Mean log magnitude per radial band.
            radial_std:
                Log-magnitude standard deviation per radial band.
            radial_power:
                Normalized power per radial band.
            spectral_entropy:
                Normalized spectral power entropy.
            spectral_centroid:
                Normalized radial power centroid.
            spectral_anisotropy:
                Relative variation in angular power distribution.
        """

        image = self._load_image(image_input)
        resized = self._resize_image(image)

        gray = cv2.cvtColor(
            resized,
            cv2.COLOR_RGB2GRAY,
        )

        gray_float = gray.astype(np.float32)

        # Remove the mean to reduce DC influence.
        centered = gray_float - float(np.mean(gray_float))

        height, width = centered.shape

        if self.apply_window:
            window_y = np.hanning(height).astype(np.float32)
            window_x = np.hanning(width).astype(np.float32)
            window = np.outer(window_y, window_x)
            signal = centered * window
        else:
            signal = centered

        spectrum = np.fft.fftshift(
            np.fft.fft2(signal)
        )

        magnitude = np.abs(spectrum).astype(np.float64)
        power = np.square(magnitude)

        log_magnitude = np.log1p(magnitude)

        geometry = self._build_frequency_geometry(
            height,
            width,
        )

        radial = self._radial_analysis(
            log_magnitude=log_magnitude,
            power=power,
            radius=geometry["radius"],
            valid_mask=geometry["valid_mask"],
            max_radius=geometry["max_radius"],
        )

        angular = self._angular_analysis(
            power=power,
            angle=geometry["angle"],
            valid_mask=geometry["valid_mask"],
        )

        spectral_entropy = self._calculate_spectral_entropy(
            power,
            geometry["valid_mask"],
        )

        heatmap = self._create_heatmap(log_magnitude)

        mean_radial_std = float(
            np.mean(radial["radial_std"])
        )

        # Preserve the historical feature-vector structure.
        feature_vector = np.asarray(
            radial["radial_mean"]
            + radial["radial_std"]
            + [radial["hf_energy_ratio"]],
            dtype=np.float32,
        )

        spectral_metrics = {
            "high_frequency_power_ratio": radial["hf_energy_ratio"],
            "total_non_dc_power": radial["total_power"],
            "high_frequency_power": radial["high_frequency_power"],
            "spectral_entropy": spectral_entropy,
            "spectral_centroid": radial["spectral_centroid"],
            "spectral_anisotropy": angular["spectral_anisotropy"],
            "peak_frequency_band": radial["peak_frequency_band"],
            "dominant_orientation_degrees": angular[
                "dominant_orientation_degrees"
            ],
            "high_frequency_start": self.high_frequency_start,
            "radial_band_count": self.num_bands,
            "angular_bin_count": self.angular_bins,
            "window_applied": self.apply_window,
            "metric_type": "descriptive_frequency_statistics",
            "is_authenticity_probability": False,
            "is_ai_detection_probability": False,
        }

        return {
            # Backward-compatible keys.
            "feature_vector": feature_vector,
            "hf_energy_ratio": radial["hf_energy_ratio"],
            "mean_variance": mean_radial_std,
            "heatmap": heatmap,
            "radial_mean": radial["radial_mean"],
            "radial_std": radial["radial_std"],

            # Extended spectral features.
            "radial_power": radial["radial_power"],
            "spectral_entropy": spectral_entropy,
            "spectral_centroid": radial["spectral_centroid"],
            "spectral_anisotropy": angular["spectral_anisotropy"],
            "angular_power_distribution": angular[
                "angular_power_distribution"
            ],
            "dominant_orientation_degrees": angular[
                "dominant_orientation_degrees"
            ],
            "peak_frequency_band": radial["peak_frequency_band"],

            # Structured diagnostics.
            "spectral_metrics": spectral_metrics,

            "analysis_metadata": {
                "input_width": int(image.shape[1]),
                "input_height": int(image.shape[0]),
                "analysis_width": width,
                "analysis_height": height,
                "color_space": "grayscale_from_rgb",
                "transform": "2D FFT",
                "spectrum_centering": "fftshift",
                "power_definition": "squared_magnitude",
                "dc_component_excluded": True,
                "window_function": (
                    "Hann" if self.apply_window else "None"
                ),
            },

            "limitations": [
                "Frequency statistics are descriptive, not proof of manipulation.",
                "High-frequency energy varies with image content, resizing and compression.",
                "Spectral entropy is not an AI-generation classifier.",
                "Angular anisotropy may reflect natural image structure.",
                "Reliable classification requires validated datasets and calibrated models.",
            ],
        }
