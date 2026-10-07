import pytest
import numpy as np
import sys
from pathlib import Path
from PIL import Image

# Add root to path
sys.path.append(str(Path(__file__).parent.parent))

from core.frequency import FrequencyEngine
from core.forensics import ForensicEngine
from core.biometrics import BiometricEngine
from core.pipeline import VeriLensPipeline

def test_frequency_engine_initialization():
    engine = FrequencyEngine(target_size=(256, 256), num_bands=64)
    assert engine.target_size == (256, 256)
    assert engine.num_bands == 64

def test_forensic_ela_engine_initialization():
    engine = ForensicEngine(quality=90, scale_factor=15.0)
    assert engine.quality == 90
    assert engine.scale_factor == 15.0

def test_biometric_engine_initialization():
    engine = BiometricEngine()
    assert engine.face_cascade is not None

def test_pipeline_end_to_end_synthetic_execution():
    pipeline = VeriLensPipeline()
    # Create test synthetic image in memory
    dummy_image = Image.new('RGB', (256, 256), color=(120, 60, 200))
    report = pipeline.analyze(dummy_image)

    assert "verdict" in report
    assert "confidence" in report
    assert "fake_probability" in report
    assert "diagnostics" in report
    assert "visualizations" in report
    assert "fft_spectrum_heatmap" in report["visualizations"]
    assert "tampering_mask" in report["visualizations"]
