
"""
VeriLens Forensics - Unified Forensic Analysis Pipeline.

Coordinates frequency analysis, Error Level Analysis (ELA), metadata
inspection, and face-related diagnostics into one structured report.

Scientific limitations:
- Heuristic scores are not calibrated probabilities.
- Metadata can be absent, altered, or forged.
- ELA and frequency features are descriptive signals, not proof.
- Low anomaly scores do not establish authenticity.
- This pipeline does not independently prove AI generation or manipulation.
"""

from typing import Any, Dict, Union

import numpy as np
from PIL import Image

from core.biometrics import BiometricEngine
from core.forensics import ForensicEngine
from core.frequency import FrequencyEngine
from core.metadata import MetadataEngine


ImageInput = Union[str, np.ndarray, Image.Image]


class VeriLensPipeline:
    """
    Run the available analysis engines and produce a cautious report.

    The legacy `fake_probability` and `confidence` fields are retained
    for compatibility, but `fake_probability` is explicitly marked as
    deprecated and `confidence` is None because no calibrated probability
    model is currently available.
    """

    def __init__(
        self,
        target_size: tuple = (256, 256),
        num_bands: int = 64,
        ela_quality: int = 90,
        ela_scale_factor: float = 15.0,
    ) -> None:
        """Initialize the forensic analysis engines."""
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

    @staticmethod
    def _finite_number(
        value: Any,
        *,
        default: float = 0.0,
        minimum: float = 0.0,
        maximum: float = 100.0,
    ) -> float:
        """
        Convert a value to a bounded finite float.

        Invalid values fall back to a supplied default. This is defensive
        normalization, not validation of the scientific meaning of a score.
        """
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError):
            return float(default)

        if not np.isfinite(number):
            return float(default)

        return float(np.clip(number, minimum, maximum))

    @staticmethod
    def _validate_image_input(image_input: ImageInput) -> None:
        """Reject unsupported or obviously invalid image inputs."""
        if isinstance(image_input, Image.Image):
            if image_input.width <= 0 or image_input.height <= 0:
                raise ValueError("Image dimensions must be positive.")
            return

        if isinstance(image_input, str):
            if not image_input.strip():
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
            "image_input must be a path string, NumPy array, or PIL Image."
        )

    @staticmethod
    def _normalize_biometric_result(result: Any) -> Dict[str, Any]:
        """Normalize the biometric engine output to a predictable structure."""
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
        normalized["biometric_risk_score"] = VeriLensPipeline._finite_number(
            result.get("biometric_risk_score", 0.0)
        )
        normalized.setdefault("biometric_verdict", "NOT_ESTABLISHED")
        normalized.setdefault("face_details", [])

        return normalized

    @staticmethod
    def _build_risk_index(
        metadata_score: float,
        ela_score: float,
        biometric_score: float,
        faces_detected: int,
    ) -> Dict[str, Any]:
        """
        Build an explicitly uncalibrated heuristic review index.

        Frequency metrics are intentionally excluded: a high-frequency
        ratio is not inherently suspicious and has no validated mapping
        to manipulation likelihood.

        The weights below are provisional engineering heuristics. They
        must not be interpreted as statistically validated weights.
        """
        if faces_detected > 0:
            score = (
                metadata_score * 0.40
                + ela_score * 0.40
                + biometric_score * 0.20
            )
            weighting_profile = "face_detected_heuristic"
        else:
            score = metadata_score * 0.50 + ela_score * 0.50
            weighting_profile = "no_face_detected_heuristic"

        score = float(np.clip(score, 0.0, 100.0))

        # These bands determine review priority only. They do not establish
        # whether a file is authentic, manipulated, or AI-generated.
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
        """
        Analyze an image and return a structured diagnostic report.

        Core engine errors are allowed to propagate instead of silently
        producing a potentially misleading partial verdict. Biometric
        analysis alone is optional and may be marked as skipped.
        """
        self._validate_image_input(image_input)

        # 1. Inspect metadata.
        meta_result = self.meta_engine.extract_metadata(image_input)

        if not isinstance(meta_result, dict):
            raise TypeError("MetadataEngine must return a dictionary.")

        # 2. Extract descriptive frequency-domain features.
        freq_result = self.freq_engine.extract_features(image_input)

        if not isinstance(freq_result, dict):
            raise TypeError("FrequencyEngine must return a dictionary.")

        # 3. Run Error Level Analysis.
        ela_result = self.ela_engine.analyze_ela(image_input)

        if not isinstance(ela_result, dict):
            raise TypeError("ForensicEngine must return a dictionary.")

        # 4. Run optional biometric analysis.
        biometric_error = None

        try:
            bio_result = self._normalize_biometric_result(
                self.bio_engine.analyze_faces(image_input)
            )
        except Exception as exc:
            # Keep the report explicit about the failed optional subsystem.
            # Do not represent a skipped biometric engine as evidence of safety.
            biometric_error = type(exc).__name__
            bio_result = {
                "faces_detected": 0,
                "biometric_verdict": "SKIPPED_OR_INCOMPATIBLE",
                "biometric_risk_score": 0.0,
                "face_details": [],
            }

        # 5. Normalize numeric outputs before constructing the report.
        # These are engine-provided heuristic scores, not calibrated
        # probabilities. The original diagnostic values are preserved below.
        metadata_score = self._finite_number(meta_result.get("risk_score", 0.0))
        ela_score = self._finite_number(ela_result.get("anomaly_score", 0.0))
        biometric_score = self._finite_number(
            bio_result.get("biometric_risk_score", 0.0)
        )

        hf_ratio = self._finite_number(
            freq_result.get("hf_energy_ratio", 0.0),
            minimum=0.0,
            maximum=1.0,
        )

        faces_detected = int(bio_result.get("faces_detected", 0))

        review = self._build_risk_index(
            metadata_score=metadata_score,
            ela_score=ela_score,
            biometric_score=biometric_score,
            faces_detected=faces_detected,
        )

        # 6. Produce a conservative verdict.
        #
        # Metadata labels alone cannot establish AI generation. If the
        # metadata engine reports a generation indicator, surface that
        # indicator without upgrading it to proof.
        metadata_verdict = str(meta_result.get("verdict", "")).upper()

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

        # 7. Preserve the established report structure for existing clients.
        #
        # `confidence` is deliberately None. `fake_probability` is retained
        # only as a deprecated compatibility alias for the review index.
        # Consumers should migrate to `risk_index`.
        report = {
            "verdict": verdict,
            "confidence": None,
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
                        "spectral_metrics", {}
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
                "review": review,
                "engine_status": {
                    "metadata": "completed",
                    "frequency": "completed",
                    "spatial_forensics": "completed",
                    "biometrics": (
                        "skipped" if biometric_error else "completed"
                    ),
                    "biometrics_error_type": biometric_error,
                },
            },
            "visualizations": {
                "fft_spectrum_heatmap": freq_result.get("heatmap"),
                "tampering_mask": ela_result.get("tampering_mask"),
            },
        }

        return report
