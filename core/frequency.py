"""
VeriLens Forensics - Core Frequency Domain Engine (v1.0)
Extracts Azimuthal Radial Averages, Intra-band Variance, and High-Frequency Ratios
via 2D Fast Fourier Transform (FFT).
"""

from typing import Dict, Any, Union
import numpy as np
import cv2


class FrequencyEngine:
    def __init__(self, target_size: tuple = (256, 256), num_bands: int = 64):
        """
        Initialize the Frequency Analysis Engine.
        :param target_size: Standard resize dimensions before computing FFT.
        :param num_bands: Number of concentric radial frequency bands.
        """
        self.target_size = target_size
        self.num_bands = num_bands

    def extract_features(self, image_input: Union[str, np.ndarray]) -> Dict[str, Any]:
        """
        Processes an image and extracts radial frequency metrics and a visual FFT heatmap.
        """
        if isinstance(image_input, str):
            img = cv2.imread(image_input)
            if img is None:
                raise ValueError(f"Could not load image from path: {image_input}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = image_input.copy()

        # Resize and convert to grayscale for uniform spectral density
        img_resized = cv2.resize(img, self.target_size)
        gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)

        # 2D Fast Fourier Transform + DC shift to matrix center
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude = 20 * np.log(np.abs(f_shift) + 1e-8)

        # Radial Euclidean distance matrix
        h, w = gray.shape
        cy, cx = h // 2, w // 2
        y, x = np.indices((h, w))
        r = np.hypot(x - cx, y - cy)

        max_radius = min(cx, cy)
        bin_width = max_radius / self.num_bands

        radial_mean = []
        radial_std = []

        for i in range(self.num_bands):
            r_start = i * bin_width
            r_end = (i + 1) * bin_width
            mask = (r >= r_start) & (r < r_end)
            if np.any(mask):
                ring = magnitude[mask]
                radial_mean.append(float(np.mean(ring)))
                radial_std.append(float(np.std(ring)))
            else:
                radial_mean.append(0.0)
                radial_std.append(0.0)

        # High-Frequency Energy Ratio (outer 40% band)
        high_freq_mask = (r >= (max_radius * 0.6)) & (r <= max_radius)
        total_mask = (r <= max_radius)
        hf_energy_ratio = float(
            np.sum(magnitude[high_freq_mask]) / (np.sum(magnitude[total_mask]) + 1e-8)
        )

        # Generate normalized visual heatmap for Web UI
        norm_mag = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        heatmap_inferno = cv2.applyColorMap(norm_mag, cv2.COLORMAP_INFERNO)
        heatmap_rgb = cv2.cvtColor(heatmap_inferno, cv2.COLOR_BGR2RGB)

        feature_vector = np.array(
            radial_mean + radial_std + [hf_energy_ratio], dtype=np.float32
        )

        return {
            "feature_vector": feature_vector,
            "hf_energy_ratio": hf_energy_ratio,
            "mean_variance": float(np.mean(radial_std)),
            "heatmap": heatmap_rgb,
      }
