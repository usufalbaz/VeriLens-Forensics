"""
VeriLens Forensics - Core Spatial & Compression Engine (v1.0)
Implements Error Level Analysis (ELA), Extrema Scaling, and Localized Tampering Masking.
"""

from typing import Dict, Any, Union
import io
import numpy as np
import cv2
from PIL import Image, ImageChops, ImageEnhance


class ForensicEngine:
    def __init__(self, quality: int = 90, scale_factor: float = 15.0):
        """
        Initialize the Error Level Analysis (ELA) Engine.
        :param quality: JPEG re-compression target quality (1-100).
        :param scale_factor: Extrema amplification multiplier for subtle discrepancies.
        """
        self.quality = quality
        self.scale_factor = scale_factor

    def analyze_ela(self, image_input: Union[str, np.ndarray, Image.Image]) -> Dict[str, Any]:
        """
        Performs Error Level Analysis to detect Photoshop splicing, copy-move, and compression anomalies.
        """
        # 1. Unify input format to PIL Image RGB
        if isinstance(image_input, str):
            original = Image.open(image_input).convert('RGB')
        elif isinstance(image_input, np.ndarray):
            original = Image.fromarray(image_input).convert('RGB')
        elif isinstance(image_input, Image.Image):
            original = image_input.convert('RGB')
        else:
            raise TypeError("Unsupported image input type for ELA processing.")

        # 2. Resave in-memory at predetermined JPEG quality
        buffer = io.BytesIO()
        original.save(buffer, 'JPEG', quality=self.quality)
        buffer.seek(0)
        resaved = Image.open(buffer)

        # 3. Compute absolute difference matrix
        difference = ImageChops.difference(original, resaved)

        # 4. Extrema amplification (Normalize difference to full 0-255 dynamic range)
        extrema = difference.getextrema()
        max_diff = max([ex[1] for ex in extrema])
        scale = 255.0 / (max_diff if max_diff != 0 else 1.0)

        enhanced = ImageEnhance.Brightness(difference).enhance(scale)
        diff_array = np.array(enhanced)

        # 5. Extract Forensic Statistical Indicators
        mean_error = float(np.mean(diff_array))
        std_error = float(np.std(diff_array))
        max_error = float(np.max(diff_array))

        # Heuristic anomaly score: Spliced regions spike localized standard deviation
        anomaly_score = float(np.clip((std_error / 50.0) * 100.0, 0.0, 100.0))

        # 6. Generate Localized Tampering Mask (JET Thermal Colormap)
        gray_diff = cv2.cvtColor(diff_array, cv2.COLOR_RGB2GRAY)
        heatmap_jet = cv2.applyColorMap(gray_diff, cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap_jet, cv2.COLOR_BGR2RGB)

        return {
            "mean_error": mean_error,
            "std_error": std_error,
            "max_error": max_error,
            "anomaly_score": anomaly_score,
            "tampering_mask": heatmap_rgb,
        }
