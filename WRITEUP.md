# Style References in Agent-Native Video Editing: Engineering Writeup

**Candidate Portfolio Project for Cardboard Internship**  
*Problem Space: Cardboard Hard Problem #6 (Style References), with direct touchpoints to #7 (Motion Graphics), #8 (Memory), and #9 (Verification).*

---

## 1. The Problem
Creators frequently want to borrow the "vibe" or editing language of a reference video—its rapid pacing, punch-in cadences, color treatment, or caption typography—and apply it to raw footage. Existing generative video models treat style transfer as end-to-end pixel hallucination, which destroys the user's content, introduces temporal flicker, and gives the creator zero editorial control.

In an agent-native browser editor like Cardboard, style transfer must be **structured, deterministic, and modular**: decompose a reference into discrete attributes, let the user choose which to transfer, apply them via the editor's rendering engine, and fall back gracefully with clear explanations when an effect or font is unavailable.

---

## 2. Technical Architecture & Key Decisions

### Decision 1: Objective Code vs. Subjective Foundation Models
Early experiments attempting to have Gemini measure shot durations or BPM produced hallucinated numbers. We established a strict architectural boundary:
- **Code measures everything objective**: PySceneDetect extracts cuts; Librosa extracts BPM and beat grids; OpenCV extracts color palettes and Farneback optical flow.
- **Gemini judges only subjective qualities**: Aesthetic grade names (`teal-orange`), font categories (`sans-serif`, `slab`), and semantic scene roles (`hook`, `body`).
Every field in the Style Card schema (`core/schema.py`) records its measurement provenance and confidence.

### Decision 2: Closed-Loop Calibration Over Open-Loop Formulae
FFmpeg color curves interact non-linearly: darkening increases HSV saturation, and warmth adjustments desaturate highlights. An open-loop formula invariably drifts. Instead, our color engine (`core/apply_engine.py`) runs a 10-step closed-loop calibration:
1. Renders a low-resolution proxy through the *identical* filter chain (including jump cuts and zoom punch-ins).
2. Measures pixel deltas against reference statistics at 2 fps.
3. Steps brightness, contrast, saturation, and warmth until convergence.

### Decision 3: The Honest Fallback Log
Nothing is silently dropped or hallucinated. If a reference uses a proprietary font or a complex whip-pan, the engine:
1. Resolves the nearest open-source alternative (via `data/fonts_catalog.json` and Google Fonts).
2. Executes a safe fallback (e.g. clean hard cut at the planned cut time).
3. Writes a structured **Fallback Log** (`core/plan.py`) detailing what was requested, what was chosen, and why.

### Decision 4: Refusing Unreliable Automated Corrections
When investigating transcript timing lag on real speech, we experimented with local cross-correlation against audio energy windows. While clean synthetic speech gave 1.0 confidence, real speech with background noise yielded low confidence (0.21) and window disagreements. Rather than applying erratic local shifts, the engine explicitly refuses auto-sync when confidence is below 0.30, recommending manual offset or transcript editing instead.

---

## 3. What Worked, What Failed, and What Was Removed

| Technique | Result | Verdict |
| :--- | :--- | :--- |
| **Beat-Snapped Jump Cuts** | Alternating framing punch-ins (1.15x) snapped to nearest beat within 300 ms. Clean editorial feel on continuous takes. | **Kept** |
| **Local Window Auto-Sync** | Attempted to shift late word clusters based on local RMS audio peaks. Often moved words onto background Foley sounds. | **Removed** |
| **Single-Frame Cut Reconciliation** | Pixel delta check overrules Gemini when soft transitions are hallucinated over hard cuts. | **Kept** |
| **Open-Loop Color Formula** | Algebraic estimation of contrast/saturation. Consistently oversaturated darkened footage. | **Replaced with Closed-Loop** |

---

## 4. Evaluation & Cardboard Roadmap Mapping

From `eval/run_eval.py` across benchmark cards:
- **Tempo Accuracy**: Librosa beat tracking achieved 0.00 BPM MAE on synthesized ground truth and ~2.5% on acoustic audio.
- **Cut Timing Precision**: Single-frame pixel cut detection matches ground-truth cuts within 0.000s MAE.
- **Label Agreement**: Gemini classification matches ground truth layout and font category at high consistency when constrained by rigid Pydantic schemas.

### Mapping to Cardboard's Core Hard Problems:
1. **#6 Style References (Core)**: Ingests reference reels into structured JSON Style Cards; applies modular bundles cleanly.
2. **#7 Motion Graphics**: Accurately classifies caption animation and kinetic typography; logs instant-swap fallbacks pending browser-canvas kinetic renderers.
3. **#8 Memory**: Style Cards act as portable, shareable editing preference assets that can be recalled across projects.
4. **#9 Verification**: Employs style-match reports and sync diagnostics to verify that rendered outputs reflect the intended reference style.
