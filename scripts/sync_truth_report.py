#!/usr/bin/env python3
"""Compare Gemini transcript word timings against human ground truth and classify timing errors.

Usage:
    python scripts/sync_truth_report.py --truth eval/sync_truth/lv_0.json --transcript outputs/lv_0.transcript.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def analyze_timing_errors(truth: list[dict[str, Any]], pred_words: list[dict[str, Any]]) -> dict[str, Any]:
    pred_map = {w["text"].strip().lower().strip(".,!?:;\"'"): float(w["start_s"]) for w in pred_words if "text" in w and "start_s" in w}

    comparisons = []
    diffs = []

    for item in truth:
        clean_text = item["text"].strip().lower().strip(".,!?:;\"'")
        true_t = float(item["true_start_s"])
        if clean_text in pred_map:
            pred_t = pred_map[clean_text]
            err = true_t - pred_t  # positive = transcript was early; negative = transcript was late
            comparisons.append({
                "word": item["text"],
                "true_start_s": true_t,
                "pred_start_s": pred_t,
                "error_s": round(err, 3),
            })
            diffs.append(err)

    if not diffs:
        return {
            "n_matched": 0,
            "comparisons": [],
            "mean_error_s": 0.0,
            "std_error_s": 0.0,
            "classification": "unknown",
            "recommendation": "No matching words found between truth and transcript.",
        }

    mean_err = sum(diffs) / len(diffs)
    variance = sum((d - mean_err) ** 2 for d in diffs) / len(diffs)
    std_err = math.sqrt(variance)

    # Classify error pattern
    if std_err <= 0.08 and abs(mean_err) >= 0.05:
        classification = "constant_offset"
        recommendation = (
            f"Constant offset detected ({mean_err:+.2f}s). "
            f"Use --caption-offset {mean_err:+.2f} when applying style."
        )
    elif len(diffs) >= 3:
        # Check for linear drift across time
        times = [c["true_start_s"] for c in comparisons]
        t_mean = sum(times) / len(times)
        cov = sum((t - t_mean) * (d - mean_err) for t, d in zip(times, diffs))
        var_t = sum((t - t_mean) ** 2 for t in times)
        slope = cov / var_t if var_t > 0 else 0.0

        if abs(slope) > 0.02:
            classification = "linear_drift"
            recommendation = (
                f"Linear drift detected (slope {slope:+.3f}s/s). "
                f"Transcript drifts over time; specify timing anchors via fix_transcript.py."
            )
        else:
            classification = "local_variance"
            recommendation = (
                "Local timing variance detected (specific words delayed or early). "
                "Use scripts/fix_transcript.py to adjust individual words."
            )
    else:
        classification = "local_variance"
        recommendation = "Local timing variance or small sample. Use scripts/fix_transcript.py."

    return {
        "n_matched": len(comparisons),
        "comparisons": comparisons,
        "mean_error_s": round(mean_err, 3),
        "std_error_s": round(std_err, 3),
        "classification": classification,
        "recommendation": recommendation,
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze caption timing errors against ground truth.")
    parser.add_argument("--truth", required=True, help="Path to ground truth JSON ([{'text': ..., 'true_start_s': ...}])")
    parser.add_argument("--transcript", required=True, help="Path to transcript JSON ([{'text': ..., 'start_s': ...}])")
    args = parser.parse_args()

    truth = json.loads(Path(args.truth).read_text(encoding="utf-8"))
    raw_tr = json.loads(Path(args.transcript).read_text(encoding="utf-8"))
    pred = raw_tr["words"] if isinstance(raw_tr, dict) else raw_tr

    report = analyze_timing_errors(truth, pred)
    print("\n=== Caption Timing Ground Truth Report ===")
    print(f"Matched {report['n_matched']} words.")
    print(f"Mean error: {report['mean_error_s']:+.3f}s (Std: {report['std_error_s']:.3f}s)")
    print(f"Classification: {report['classification']}")
    print(f"Recommendation: {report['recommendation']}\n")

    print(f"{'Word':<15} {'True Start':>10} {'Pred Start':>10} {'Error (s)':>10}")
    print("-" * 50)
    for c in report["comparisons"]:
        print(f"{c['word']:<15} {c['true_start_s']:>9.2f}s {c['pred_start_s']:>9.2f}s {c['error_s']:>+9.3f}s")


if __name__ == "__main__":
    main()
