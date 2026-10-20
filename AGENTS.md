# AGENTS.md: Style Transfer Planner

Read this first. It is the standing context for any coding agent working in this repo. The ordered work list is in
`CODEX_TASKS.md`. The human owner is a CS student building this as a portfolio demo for an internship application at
**Cardboard** (a browser-based, agent-native video editor). Their published "hard problem #6" is **Style References**:
ingest a reference video, break it into discrete actionable attributes, let the user choose which to apply, reproduce
them reliably, and fall back gracefully when something (a font, an effect) isn't available.

## What the product does
1. **Analyze** a short reference video into a structured **Style Card** (JSON): pacing, cuts, BPM/beat sync, color
   statistics and grade, text/caption style, transitions, shot types, layout, mood, plus scene-by-scene data and
   *coupling* metrics (how attributes relate). Code measures anything objective; Gemini judges only subjective things.
2. **Apply** selected attributes to a target clip with FFmpeg: color match, jump-cut pacing (beat-snapped), transitions
   (hard cut / zoom punch-in rendered), burned-in captions in the reference's font/size/color/position/box. Returns a
   plan, a **fallback log** (what could not be copied and what was used instead), and a style-match report.
3. **Streamlit app** (`app.py`) wraps both. CLIs exist for every step.

## Status (be honest about this in anything you write)
**Done and tested (180 offline tests):** Stage 1 analysis pipeline; Stage 2 apply engine; font resolver + bundled fonts;
captions via Gemini transcription or user JSON; sync-check diagnostics; debug overlays; Streamlit app with 4 tabs.
**Verified by the owner on real clips:** analysis of three different reels, caption style/position match, Gemini
transcription through the UI, color calibration closing the loop on a real clip.
**NOT done / open:**
- Stage 3: evaluation harness, usage cap, public deployment, pre-cached demo cards, write-up/demo script/outreach.
- **Open bug:** on one real clip (`lv_0_20231212164716.mp4`) the owner perceives captions as late. Rendering was
  tested for delay and found clean on synthetic clips; the clip is constant-frame-rate; `check_sync` was inconclusive on
  real speech (confidence 0.21, windows disagreed). Root cause unknown. Do not claim it is fixed without human confirmation.
- Not rendered: text animation (pop/scale etc.), transitions other than hard cut / zoom punch-in, split-screen or
  picture-in-picture layout transfer, per-scene typography for kinetic-text references.
- Live Gemini behaviour through the *Analyze new reference* button in the app has only been tested with a fake client.

## Repository map
```
app.py                    Streamlit UI (tabs: Inputs, Style card, Choose what to borrow, Result)
core/
  utils.py                errors (AnalysisError, InvalidVideoError, VideoTooLongError), MAX_DURATION_S=60, ffprobe_info, run(), temp_workdir
  schema.py               Pydantic StyleCard + Gemini response models; vocabularies; normalize_choice; SCHEMA_VERSION ("1.2")
  frames.py               FrameReader (seek + iter_frames), jpeg/montage helpers
  scenes.py               PySceneDetect wrapper, merge_short_spans, pacing_stats, scene energy (relative labels)
  audio.py                librosa BPM/beats/peaks/energy, beat_sync_score
  color.py                palette (k-means), frame_stats, contrast/saturation/warmth labels, heuristic grade
  motion.py               optical-flow magnitude + camera hint per scene
  gemini_client.py        GeminiClient: pacing, retries, error classification, fatal short-circuit, JSON validation, usage log
  prompts.py              ALL prompts (named constants)
  mock_gemini.py          offline fake backend (clearly marked MOCK; cache goes to data/cache/mock)
  transitions.py          measure_cut (single-frame test), Gemini classification, reconcile(label, abrupt)
  text_style.py           text presence + style via Gemini, TextDetection (region/scattered), caption_activity (measured)
  coupling.py             relationships between attributes (beat sync, text~pacing, grade consistency, notes)
  pipeline.py             analyze_video(): orchestrates everything, caches data/cache/<sha256>.json (degraded results never cached)
  plan.py                 EditingPlan, PlanItem, FallbackEntry, StyleWarning, StyleMatch, ApplyResult
  fonts.py                catalog (data/fonts_catalog.json, 53 Google Fonts) + attribute-based resolver (close/fallback)
  font_files.py           font FILE chain: local data/fonts (18 bundled) -> download (Google Fonts on GitHub) -> system -> Pillow default
  bundles.py              attribute bundles + data-driven dependency warnings
  apply_engine.py         apply_style(): color_adjust, calibrate_pipeline (closed loop on the REAL cuts/zoom/filter chain),
                          plan_cuts, segment_zooms, render (ffmpeg filter_complex), captions, style_match
  captions.py             chunking, PIL overlay compositing frame-by-frame, variable-frame-rate normalisation
  transcript.py           Gemini audio transcription with word timestamps, clean_words, JSON loader
  sync_check.py           transcript-vs-audio GLOBAL lag estimator (diagnostic; local auto-correction was tried and rejected)
  debug_overlay.py        labelled debug video + before/after strip per cut
  ui_helpers.py           pure helpers for the app (dirs via STP_* env vars, tables, charts, previews)
scripts/                  analyze_cli, apply_cli, check_gemini, check_sync, fix_transcript, plot_sync, preview_card, make_synthetic
tests/                    pytest suite (offline). conftest.py builds synthetic videos with KNOWN ground truth
data/fonts/               bundled OFL fonts (+ LICENSES.md);  data/fonts_catalog.json
data/examples/            the owner's local test videos (NOT committed; contains other creators' content)
data/cache/               analysis cache (gitignored in practice via outputs/; mock results in data/cache/mock)
outputs/                  generated cards, renders, uploads (gitignored)
```
Key behaviour entry points: `analyze_video(path, client=None, cache_dir=None, use_cache=True, progress=None)` and
`apply_style(card, target, selected, out_path, strength, zoom, client, transcript, caption_offset_s, words_per_caption, auto_sync)`.
Attributes: `pacing, audio_rhythm, transitions, color, text`.

## Commands
Windows PowerShell (the owner's platform; Python 3.14 in a venv named `.venv` one folder ABOVE the project):
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
..\.venv\Scripts\Activate.ps1                      # or: python -m venv .venv ; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m pytest -q                                # ~3 min, 180 tests, no network, no API key needed
python -m streamlit run app.py                     # http://localhost:8501
python scripts/check_gemini.py [--model NAME] [--models]
python scripts/analyze_cli.py "path with spaces.mp4" [--no-cache] [--mock-gemini]
python scripts/apply_cli.py --reference outputs/X.stylecard.json --target clip.mp4 --apply pacing,audio_rhythm,transitions,color,text --out outputs/o.mp4 [--debug] [--transcript words.json] [--auto-sync]
```
Linux/macOS equivalents are identical minus the activation line. FFmpeg and ffprobe must be on PATH (>= 5.1 for `-fps_mode`).
**Quote every path** (the owner's paths contain spaces). Prefer `python -m ...` over bare `streamlit`/`pytest`.

## Environment and secrets
- `.env` (gitignored) is loaded by the CLIs/app: `GEMINI_API_KEY`, optional `GEMINI_MODEL`, `GEMINI_MIN_INTERVAL_S` (default 5),
  `GEMINI_MAX_RETRIES` (default 3). Blank or invalid numeric values are ignored. Directory overrides: `STP_CARDS_DIR`,
  `STP_VIDEOS_DIR`, `STP_OUT_DIR`.
- **Never** print, log, commit, screenshot, or paste the API key. Never write it into any tracked file, including `.env.example`.
  The owner's earlier key was exposed in screenshots; assume they rotate it.
- The working model on the owner's key is `gemini-3.5-flash-lite` (set via `GEMINI_MODEL`). The code default
  `gemini-flash-latest` was overloaded/over quota; `gemini-2.5-flash-lite` is retired for new users. Models change: run
  `python scripts/check_gemini.py --models` and `--model NAME` (text AND image call) before assuming a name works.
- Free-tier Gemini has per-minute and per-day quotas; every retry counts. A full reference analysis is ~7-10 requests, captions 1.
- **Tests must never call a real API.** Use `GeminiClient(backend=fake)` or `mock_backend`.

## Engineering rules (follow these)
1. Every behaviour change gets a test. Run the full suite before you finish a task. Keep tests offline and fast; synthetic
   videos from `scripts/make_synthetic.py` give known ground truth (BPM, cut times, captions) so assert against truth, not against
   your own output.
2. **Honesty over polish.** Each attribute records whether it was measured by code or judged by Gemini, plus a confidence that is
   heuristic/self-reported, NOT calibrated. Do not add language that implies accuracy that was not measured. Anything that
   cannot be reproduced goes in the fallback log with a reason; nothing may silently fall back.
3. Fail soft: a failed Gemini call degrades that attribute (null + confidence 0 + `meta.degraded`), never crashes the pipeline.
   Degraded results are not cached. Fatal API errors (bad key, unknown model, daily quota) short-circuit later calls.
4. If you change the shape of the Style Card, bump `SCHEMA_VERSION` in `core/schema.py` (old cached cards are then re-analysed).
5. Keep prompts in `core/prompts.py`; normalise model output onto fixed vocabularies with `normalize_choice`.
6. Do not rewrite working modules. Make the smallest change that meets the acceptance criteria. No new heavy dependencies
   (torch, tensorflow, etc.) without asking; optional extras must be lazily imported and the feature must work without them.
7. Never commit videos, `.env`, outputs, caches, or API keys. Other creators' videos live in `data/examples/` only on the owner's disk.
8. Update the README's "Verified vs not" language whenever you add or remove a claim.
9. Commit after each task with a clear message. Do not rewrite git history.
10. When done with a task, report in this format: **Changed** (files), **Tests** (count + result), **Verified by running** (what you
    actually executed and saw), **NOT verified** (what needs a human or a live service), **Follow-ups**.

## Hard-won lessons (do not relearn these)
- FFmpeg `scale` silently changes the pixel aspect ratio; `concat` rejects mixed SAR. Every segment needs `setsar=1`; crop to even sizes.
- Variable-frame-rate sources must be converted to constant frame rate before any frame-by-frame pass, or the picture time-warps
  against the untouched audio (`captions.ensure_constant_frame_rate`). The owner's current clip is constant (30 fps).
- Color calibration must render the SAME pipeline (cuts, zooms, filters) it will ship, and measure EVERY frame of the probe at 2 fps.
  Sampling a 1 fps probe at second midpoints aliases with 1-second cut cadences (it silently saw only unzoomed segments).
- HSV saturation rises when you darken; warmth `colorbalance` desaturates. Calibrate jointly with a measured loop (10 steps), not a formula.
- PySceneDetect misses nothing on hard cuts but ignores soft ones; `measure_cut` uses a single-frame test and OVERRULES Gemini when it
  contradicts the pixels (`transitions.reconcile`). Gemini's per-run labels vary; temperature 0 + seed reduces, not removes, that.
- A scene cut can be a change in ONE PANEL of a composite layout (screen recording above a facecam). Whole-frame camera motion is meaningless
  there; screen-recording shots are forced to `static`.
- Gemini returns 0-1000 or 0-1 boxes inconsistently: `text_style.normalize_bbox` accepts both. Scattered boxes mean kinetic typography:
  no single caption region, `position="mixed"`, caption pacing not measured.
- Estimating transcript lag from audio energy works on clean synthetic speech and is unreliable on real speech. Trust requires
  confidence >= 0.30, windows that agree, and a lag away from the edge of the search range. Local per-window auto-correction was tried
  and made things worse in some cases: do not reintroduce it without real-speech ground truth.
- Gemini errors: 503 = transient overload (retry with backoff), 429 per-minute = retry after the hinted delay, 429 per-day / 401 / 403 /
  404 = fatal for the run. Calls are paced `GEMINI_MIN_INTERVAL_S` apart. Retries count toward quota.
- Windows: paths contain spaces; PowerShell needs quotes; use `python -m`; VS Code terminals may not have the venv active; `source` does not exist.
- Streamlit: widget keys cannot be modified after instantiation (use a pending-state key applied before the widget, or a callback);
  `use_container_width` is deprecated in favour of `width="stretch"` (helpers in `app.py` fall back); AppTest cannot upload files.
- Scene-detector counts on rendered outputs undercount small-framing jump cuts; the style-match report says so.

## Out of scope unless the owner asks
Rebuilding the UI in another framework; training models; storing user uploads permanently; scraping or downloading videos from social
platforms; claiming accuracy numbers that were not measured.

## Where to look next
`CODEX_TASKS.md` (ordered tasks, acceptance criteria, prompts) and `README.md` (usage, verified vs not).
