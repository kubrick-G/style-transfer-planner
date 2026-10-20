# Evaluation Results: Style Transfer Planner

**Sample Size**: 4 clips evaluated.
> [!NOTE]
> n=4: indicative only (small sample size). Results should be interpreted as trend indicators.

## 1. Ground Truth Agreement

| Attribute | Accuracy | Correct / Total | Measurement Provenance |
| :--- | :--- | :--- | :--- |
| Layout | 100.0% | 2/2 | Gemini (Subjective) |
| Color Grade | 100.0% | 4/4 | Gemini + Pixel Statistics |
| Caption Position | 100.0% | 4/4 | Code Bounding Box + Gemini |
| Text Animation | 100.0% | 4/4 | Gemini |
| Font Category | 100.0% | 4/4 | Gemini + Font Catalog |

## 2. Objective Numeric Metrics

- **Tempo MAE**: 0.00 BPM (audio beat detection via Librosa)
- **Cut Timing MAE**: 0.000 s (single-frame Pixel transitions)
- **Cut Recall**: 100.0% (within 0.20s window)

## 3. Color Grade Confusion Summary

| True Grade | Predicted Grade | Count |
| :--- | :--- | :--- |
| natural | natural | 3 |
| other | other | 1 |
