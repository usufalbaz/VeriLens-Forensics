
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.frequency import FrequencyEngine
from core.forensics import ForensicEngine
from core.biometrics import BiometricEngine
from core.metadata import MetadataEngine
from core.pipeline import VeriLensPipeline


@pytest.fixture
def sample_image():
    return Image.new("RGB", (256, 256), color=(120, 60, 200))


def test_frequency_engine_initialization():
    engine = FrequencyEngine(target_size=(256, 256), num_bands=64)

    assert engine.target_size == (256, 256)
    assert engine.num_bands == 64


def test_frequency_engine_returns_valid_features(sample_image):
    result = FrequencyEngine().extract_features(sample_image)

    assert "hf_energy_ratio" in result
    assert np.isfinite(result["hf_energy_ratio"])
    assert "heatmap" in result
    assert result["heatmap"].shape[:2] == (256, 256)


def test_forensic_engine_initialization():
    engine = ForensicEngine(quality=90, scale_factor=15.0)

    assert engine.quality == 90
    assert engine.scale_factor == 15.0


def test_forensic_engine_returns_valid_ela_metrics(sample_image):
    result = ForensicEngine().analyze_ela(sample_image)

    for key in ("mean_error", "std_error", "max_error", "anomaly_score"):
        assert key in result
        assert np.isfinite(result[key])

    assert 0 <= result["anomaly_score"] <= 100
    assert result["tampering_mask"].shape == (256, 256, 3)
    assert "limitations" in result


def test_forensic_engine_rejects_invalid_quality():
    with pytest.raises(ValueError):
        ForensicEngine(quality=0)


def test_biometric_engine_initializes():
    engine = BiometricEngine()

    assert hasattr(engine, "face_cascade")


def test_biometric_engine_does_not_claim_authenticity(sample_image):
    result = BiometricEngine().analyze_faces(sample_image)

    assert "faces_detected" in result
    assert "face_details" in result
    assert "limitations" in result
    assert result["biometric_verdict"] in {
        "INCONCLUSIVE",
        "MODULE_UNAVAILABLE",
        "FACE_DETECTED_INCONCLUSIVE",
        "NO_FACES_DETECTED",
    }


def test_metadata_engine_returns_limited_indicators(sample_image):
    result = MetadataEngine().extract_metadata(sample_image)

    assert "risk_score" in result
    assert "verdict" in result
    assert "limitations" in result
    assert "analysis_scope" in result


def test_pipeline_end_to_end(sample_image):
    report = VeriLensPipeline().analyze(sample_image)

    assert "verdict" in report
    assert "confidence" in report
    assert "fake_probability" in report
    assert "diagnostics" in report
    assert "visualizations" in report

    visualizations = report["visualizations"]
    assert "fft_spectrum_heatmap" in visualizations
    assert "tampering_mask" in visualizations

    assert visualizations["fft_spectrum_heatmap"].size > 0
    assert visualizations["tampering_mask"].size > 0
