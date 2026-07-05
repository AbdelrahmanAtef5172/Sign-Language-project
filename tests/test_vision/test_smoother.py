import pytest
from vision.inference.temporal_smoother import TemporalSmoother, PairedSmoother

def test_temporal_smoother_majority_vote():
    """Verify smoother returns the majority vote."""
    smoother = TemporalSmoother(window_size=5)
    for val in ["male", "male", "female", "male", "female"]:
        smoother.add_prediction(val)
    
    assert smoother.get_stable_prediction() == "male"

def test_paired_smoother_averaging():
    """Verify PairedSmoother averages confidence."""
    smoother = PairedSmoother(window_size=2)
    smoother.add("happy", 0.8)
    smoother.add("happy", 0.4)
    
    label, conf = smoother.get()
    assert label == "happy"
    assert conf == pytest.approx(0.6)

def test_smoother_clear():
    """Verify clear() resets the smoother."""
    smoother = TemporalSmoother(window_size=5)
    smoother.add_prediction("male")
    smoother.clear()
    assert smoother.get_stable_prediction() is None
