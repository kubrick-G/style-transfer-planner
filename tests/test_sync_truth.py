import pytest
from scripts.sync_truth_report import analyze_timing_errors


def test_constant_offset_classification():
    truth = [
        {"text": "one", "true_start_s": 1.25},
        {"text": "two", "true_start_s": 2.24},
        {"text": "three", "true_start_s": 3.26},
    ]
    pred = [
        {"text": "one", "start_s": 1.0},
        {"text": "two", "start_s": 2.0},
        {"text": "three", "start_s": 3.0},
    ]
    report = analyze_timing_errors(truth, pred)
    assert report["n_matched"] == 3
    assert report["classification"] == "constant_offset"
    assert "--caption-offset" in report["recommendation"]
    assert abs(report["mean_error_s"] - 0.25) < 0.02


def test_linear_drift_classification():
    truth = [
        {"text": "start", "true_start_s": 1.0},
        {"text": "middle", "true_start_s": 5.2},
        {"text": "end", "true_start_s": 10.5},
    ]
    pred = [
        {"text": "start", "start_s": 1.0},
        {"text": "middle", "start_s": 5.0},
        {"text": "end", "start_s": 10.0},
    ]
    report = analyze_timing_errors(truth, pred)
    assert report["n_matched"] == 3
    assert report["classification"] == "linear_drift"
    assert "Linear drift" in report["recommendation"]
