
"""
VeriLens Forensics - Unified Forensic Analysis Pipeline.

Coordinates metadata inspection, frequency analysis, Error Level
Analysis, biometric diagnostics, and optional deepfake inference.

Scientific limitations:
- Heuristic scores are not calibrated probabilities.
- Deepfake predictions depend on the model and its training data.
- Metadata can be absent, altered, or forged.
- ELA and frequency features are descriptive signals, not proof.
- Low anomaly scores do not establish authenticity.
- No single engine independently proves image authenticity.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
from PIL import Image

from core.biometrics import BiometricEngine
from core.deepfake import DeepfakeEngine
from core.forensics import ForensicEngine
from core.frequency import FrequencyEngine
from core.metadata import MetadataEngine


ImageInput = Union[str, Path, np.ndarray, Image.Image]


class VeriLensPipeline:
    """
    Unified forensic analysis pipeline.

    Deepfake inference is optional. Its output is reported independently
    and is not included in the heuristic risk index.
    """

    def __init__(
        self,
        target_size: tuple = (256, 256),
        num_bands: int = 64,
        ela_quality: int = 90,
        ela_scale_factor: float = 15.0,
        deepfake_model_path: Optional[str] = None,
        deepfake_threshold: float = 0.5,
    ) -> None:

        self.freq_engine = FrequencyEngine(
            target_size=target_size,
            num_bands=num_bands,
        )

        self.ela_engine = ForensicEngine(
            quality=ela_quality,
            scale_factor=ela_scale_factor,
        )

        self.meta_engine = MetadataEngine()
        self.bio_engine = BiometricEngine()

        self.deepfake_engine = DeepfakeEngine(
            model_path=deepfake_model_path,
            threshold=deepfake_threshold,
        )

    @staticmethod
    def _finite_number(
        value: Any,
        *,
        default: float = 0.0,
        minimum: float = 0.0,
        maximum: float = 100.0,
    ) -> float:

        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError):
            return float(default)

        if not np.isfinite(number):
            return float(default)

        return float(np.clip(number, minimum, maximum))

    @staticmethod
    def _validate_image_input(image_input: ImageInput) -> None:

        if isinstance(image_input, Image.Image):
            if image_input.width <= 0 or image_input.height <= 0:
                raise ValueError("Image dimensions must be positive.")
            return

        if isinstance(image_input, (str, Path)):
            if not str(image_input).strip():
                raise ValueError("Image path must not be empty.")
            return

        if isinstance(image_input, np.ndarray):
            if image_input.size == 0:
                raise ValueError("Image array must not be empty.")

            if image_input.ndim not in (2, 3):
                raise ValueError(
                    "Image array must be grayscale or a color image."
                )

            if image_input.ndim == 3 and image_input.shape[2] not in (3, 4):
                raise ValueError(
                    "Color image arrays must have 3 (RGB) or 4 (RGBA) channels."
                )

            if image_input.shape[0] <= 0 or image_input.shape[1] <= 0:
                raise ValueError("Image dimensions must be positive.")

            return

        raise TypeError(
            "image_input must be a path, NumPy array, or PIL Image."
        )

    @staticmethod
    def _normalize_biometric_result(result: Any) -> Dict[str, Any]:

        if not isinstance(result, dict):
            raise TypeError("BiometricEngine must return a dictionary.")

        faces_detected = result.get("faces_detected", 0)

        if (
            isinstance(faces_detected, bool)
            or not isinstance(faces_detected, (int, np.integer))
            or faces_detected < 0
        ):
            faces_detected = 0

        normalized = dict(result)
        normalized["faces_detected"] = int(faces_detected)

        normalized["biometric_risk_score"] = (
            VeriLensPipeline._finite_number(
                result.get("biometric_risk_score", 0.0)
            )
        )

        normalized.setdefault(
            "biometric_verdict",
            "NOT_ESTABLISHED",
        )
        normalized.setdefault("face_details", [])

        return normalized

    @staticmethod
    def _build_risk_index(
        metadata_score: float,
        ela_score: float,
        biometric_score: float,
        faces_detected: int,
    ) -> Dict[str, Any]:

        if faces_detected > 0:
            score = (
                metadata_score * 0.40
                + ela_score * 0.40
                + biometric_score * 0.20
            )
            weighting_profile = "face_detected_heuristic"

        else:
            score = (
                metadata_score * 0.50
                + ela_score * 0.50
            )
            weighting_profile = "no_face_detected_heuristic"

        score = float(np.clip(score, 0.0, 100.0))

        if score >= 65.0:
            review_status = "HIGH_REVIEW_PRIORITY"
        elif score >= 35.0:
            review_status = "REQUIRES_REVIEW"
        else:
            review_status = "NO_STRONG_HEURISTIC_SIGNAL"

        return {
            "risk_index": round(score, 2),
            "review_status": review_status,
            "weighting_profile": weighting_profile,
            "is_calibrated_probability": False,
            "thresholds_validated": False,
            "interpretation": (
                "A heuristic prioritization score only. It is not a "
                "probability of manipulation and cannot establish authenticity."
            ),
        }

    def analyze(self, image_input: ImageInput) -> Dict[str, Any]:

        self._validate_image_input(image_input)

        # 1. Metadata inspection.
        meta_result = self.meta_engine.extract_metadata(image_input)

        if not isinstance(meta_result, dict):
            raise TypeError("MetadataEngine must return a dictionary.")

        # 2. Frequency-domain analysis.
        freq_result = self.freq_engine.extract_features(image_input)

        if not isinstance(freq_result, dict):
            raise TypeError("FrequencyEngine must return a dictionary.")

        # 3. Error Level Analysis.
        ela_result = self.ela_engine.analyze_ela(image_input)

        if not isinstance(ela_result, dict):
            raise TypeError("ForensicEngine must return a dictionary.")

        # 4. Optional biometric analysis.
        biometric_error = None

        try:
            bio_result = self._normalize_biometric_result(
                self.bio_engine.analyze_faces(image_input)
            )

        except Exception as exc:
            biometric_error = type(exc).__name__

            bio_result = {
                "faces_detected": 0,
                "biometric_verdict": "SKIPPED_OR_INCOMPATIBLE",
                "biometric_risk_score": 0.0,
                "face_details": [],
            }

        # 5. Optional deepfake inference.
        # An unavailable model must not prevent other engines from running.
        deepfake_result = self.deepfake_engine.analyze(image_input)

        if not isinstance(deepfake_result, dict):
            deepfake_result = {
                "engine": "deepfake",
                "status": "error",
                "available": False,
                "prediction": None,
                "fake_probability": None,
                "real_probability": None,
                "error": "DeepfakeEngine returned an invalid result.",
            }

        deepfake_status = deepfake_result.get("status", "unknown")

        # 6. Normalize heuristic outputs.
        metadata_score = self._finite_number(
            meta_result.get("risk_score", 0.0)
        )

        ela_score = self._finite_number(
            ela_result.get("anomaly_score", 0.0)
        )

        biometric_score = self._finite_number(
            bio_result.get("biometric_risk_score", 0.0)
        )

        hf_ratio = self._finite_number(
            freq_result.get("hf_energy_ratio", 0.0),
            minimum=0.0,
            maximum=1.0,
        )

        faces_detected = int(
            bio_result.get("faces_detected", 0)
        )

        review = self._build_risk_index(
            metadata_score=metadata_score,
            ela_score=ela_score,
            biometric_score=biometric_score,
            faces_detected=faces_detected,
        )

        # 7. Conservative heuristic verdict.
        metadata_verdict = str(
            meta_result.get("verdict", "")
        ).upper()

        if metadata_verdict == "AI_GENERATION_CONFIRMED":
            verdict = "AI_GENERATION_INDICATOR_PRESENT"
            verdict_basis = (
                "The metadata engine reported a generation indicator. "
                "This is not independent verification of image origin."
            )

        elif review["review_status"] == "HIGH_REVIEW_PRIORITY":
            verdict = "REQUIRES_FORENSIC_REVIEW"
            verdict_basis = (
                "The provisional heuristic index exceeded the review "
                "threshold. This does not prove manipulation."
            )

        elif review["review_status"] == "REQUIRES_REVIEW":
            verdict = "SUSPICIOUS_INCONCLUSIVE"
            verdict_basis = (
                "Available heuristic signals warrant review but do not "
                "establish manipulation or AI generation."
            )

        else:
            verdict = "NO_STRONG_HEURISTIC_SIGNAL"
            verdict_basis = (
                "The current heuristic signals are limited. This does not "
                "establish that the image is authentic or unmodified."
            )

        # 8. Build the unified report.
        report = {
            "verdict": verdict,
            "confidence": None,

            # Deprecated compatibility field.
            "fake_probability": round(review["risk_index"], 2),

            "risk_index": review["risk_index"],
            "score_type": "uncalibrated_heuristic_review_index",
            "verdict_basis": verdict_basis,

            "limitations": [
                "No calibrated manipulation probability is available.",
                "A low risk index does not prove authenticity.",
                "Metadata indicators do not independently prove image origin.",
                "Frequency and ELA features are not standalone proof of tampering.",
                "Biometric analysis may be unavailable or inconclusive.",
                "Deepfake predictions depend on the model and its training data.",
                "An unavailable deepfake model does not imply that an image is real.",
                "The deepfake output is not included in the heuristic risk index.",
            ],

            "diagnostics": {
                "metadata": meta_result,

                "frequency": {
                    "high_frequency_ratio": round(hf_ratio, 6),
                    "mean_spectral_variance": self._finite_number(
                        freq_result.get("mean_variance", 0.0),
                        minimum=0.0,
                        maximum=1_000_000.0,
                    ),
                    "spectral_metrics": freq_result.get(
                        "spectral_metrics",
                        {},
                    ),
                },

                "spatial_forensics": {
                    "mean_error": self._finite_number(
                        ela_result.get("mean_error", 0.0),
                        minimum=0.0,
                        maximum=1_000_000.0,
                    ),
                    "std_error": self._finite_number(
                        ela_result.get("std_error", 0.0),
                        minimum=0.0,
                        maximum=1_000_000.0,
                    ),
                    "tampering_risk": ela_score,
                    "interpretation": (
                        "ELA is a diagnostic signal and is not proof "
                        "of manipulation."
                    ),
                },

                "biometrics": bio_result,

                "deepfake": deepfake_result,

                "review": review,

                "engine_status": {
                    "metadata": "completed",
                    "frequency": "completed",
                    "spatial_forensics": "completed",
                    "biometrics": (
                        "skipped" if biometric_error else "completed"
                    ),
                    "biometrics_error_type": biometric_error,
                    "deepfake": deepfake_status,
                },
            },

            "visualizations": {
                "fft_spectrum_heatmap": freq_result.get("heatmap"),
                "tampering_mask": ela_result.get("tampering_mask"),
            },
        }

        return report
