"""
VeriLens Forensics - Unified Forensic Pipeline (v1.0)
Orchestrates Frequency Analysis, Error Level Analysis (ELA),
Metadata Provenance, and Biometric Landmark Verification into a consolidated report.
"""

from typing import Dict, Any, Union
import numpy as np
from PIL import Image

from core.frequency import FrequencyEngine
from core.forensics import ForensicEngine
from core.metadata import MetadataEngine
from core.biometrics import BiometricEngine


class VeriLensPipeline:
    def __init__(self):
        """
        Initialize all forensic engines within a single unified pipeline.
        """
        self.freq_engine = FrequencyEngine(target_size=(256, 256), num_bands=64)
        self.ela_engine = ForensicEngine(quality=90, scale_factor=15.0)
        self.meta_engine = MetadataEngine()
        self.bio_engine = BiometricEngine()

    def analyze(self, image_input: Union[str, np.ndarray, Image.Image]) -> Dict[str, Any]:
        """
        Executes multi-layered forensic inspection on the provided image.
        Returns a comprehensive diagnostic dictionary.
        """
        # 1. Execute Metadata Inspection
        meta_result = self.meta_engine.extract_metadata(image_input)

        # 2. Execute Frequency Domain Analysis (2D-FFT)
        freq_result = self.freq_engine.extract_features(image_input)

        # 3. Execute Spatial Error Level Analysis (ELA)
        ela_result = self.ela_engine.analyze_ela(image_input)

        # 4. Execute Biometric Facial & Corneal Reflection Verification
        try:
            bio_result = self.bio_engine.analyze_faces(image_input)
        except Exception:
            bio_result = {
                "faces_detected": 0,
                "biometric_verdict": "SKIPPED_OR_INCOMPATIBLE",
                "biometric_risk_score": 0.0,
                "face_details": []
            }

        # 5. Synthesize Forensic Decision Logic
        spectral_score = float(np.clip(freq_result["hf_energy_ratio"] * 100.0, 0.0, 100.0))
        ela_score = float(ela_result["anomaly_score"])
        meta_score = float(meta_result["risk_score"])
        bio_score = float(bio_result["biometric_risk_score"])

        # Dynamic weighting based on face detection
        if bio_result["faces_detected"] > 0:
            composite_fake_probability = (
                (spectral_score * 0.30) +
                (ela_score * 0.30) +
                (meta_score * 0.20) +
                (bio_score * 0.20)
            )
        else:
            composite_fake_probability = (
                (spectral_score * 0.40) +
                (ela_score * 0.35) +
                (meta_score * 0.25)
            )

        composite_fake_probability = float(np.clip(composite_fake_probability, 0.0, 100.0))

        # Overall Forensic Verdict Determination
        if meta_result["verdict"] == "AI_GENERATION_CONFIRMED":
            final_verdict = "AI_GENERATED_CONFIRMED"
            confidence = 99.0
        elif composite_fake_probability >= 65.0:
            final_verdict = "HIGH_MANIPULATION_PROBABILITY"
            confidence = composite_fake_probability
        elif composite_fake_probability <= 35.0:
            final_verdict = "AUTHENTIC_ORGANIC_MEDIA"
            confidence = 100.0 - composite_fake_probability
        else:
            final_verdict = "SUSPICIOUS_INCONCLUSIVE"
            confidence = 50.0

        return {
            "verdict": final_verdict,
            "confidence": round(confidence, 2),
            "fake_probability": round(composite_fake_probability, 2),
            "diagnostics": {
                "metadata": meta_result,
                "frequency": {
                    "high_frequency_ratio": round(freq_result["hf_energy_ratio"], 4),
                    "mean_spectral_variance": round(freq_result["mean_variance"], 3)
                },
                "spatial_forensics": {
                    "mean_error": round(ela_result["mean_error"], 2),
                    "std_error": round(ela_result["std_error"], 2),
                    "tampering_risk": round(ela_result["anomaly_score"], 2)
                },
                "biometrics": bio_result
            },
            "visualizations": {
                "fft_spectrum_heatmap": freq_result["heatmap"],
                "tampering_mask": ela_result["tampering_mask"]
            }
        }
