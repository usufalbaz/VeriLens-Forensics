import pytest
import numpy as np
import sys
from pathlib import Path

# Add root to path
sys.path.append(str(Path(__file__).parent.parent))

from core.frequency import FrequencyEngine
from core.forensics import ForensicEngine
from core.pipeline import VeriLensPipeline

def test_frequency_engine_initialization():
    engine = FrequencyEngine(target_size=(256, 256), num_bands=64)
    assert engine.target_size == (256, 256)
    assert engine.num_bands == 64

def test_forensic_ela_engine_initialization():
    engine = ForensicEngine(quality=90, scale_factor=15.0)
    assert engine.quality == 90
    assert engine.scale_factor == 15.0

def test_pipeline_orchestrator_loads():
    pipeline = VeriLensPipeline()
    assert pipeline.freq_engine is not None
    assert pipeline.ela_engine is not None
    assert pipeline.meta_engine is not None
