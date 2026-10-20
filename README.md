# Style Transfer Planner — Stage 1: analysis pipeline

Extracts a short vertical video's **editing style** into a structured, machine-readable **Style Card** (JSON):
pacing, color grade, text style, transitions, audio rhythm, camera/shot types, per-scene attributes, and
**coupling** (how attributes relate — a style's "feel" lives in the combination, e.g. cuts landing on the beat).

Stages 2-3 (apply engine, Streamlit UI, evaluation, deployment) build on `core/schema.py` and `core/pipeline.py`.

## Principle
**Code measures anything objective (numbers/timing); Gemini only makes subjective judgments.** Every card records
which is which (`provenance`) plus a per-attribute `confidence` (heuristic / self-reported, *not calibrated*).

| Measured by code | Judged by Gemini |
|---|---|
| shot boundaries, cuts/min, pacing curve | color grade *name* (told the measured stats; may not contradict them) |
| BPM, beat times, beat-sync score, music peaks, energy | film grain |
| palette, saturation/contrast/warmth stats, vignette | shot type, camera-motion label (given optical-flow hint) |
| optical-flow motion magnitude + camera hint | transition *type* (code pre-labels hard cut vs other) |
| scene energy, coupling metrics | text presence + style (as attributes, never a font name), animation |
| | mood, genre, hook/body/end sanity check |

## Setup
```bash
pip install -r requirements.txt          # also needs FFmpeg/ffprobe on PATH (see packages.txt)
cp .env.example .env                     # put your GEMINI_API_KEY in .env (never commit it)
```

## Use
```bash
python scripts/analyze_cli.py my_reference.mp4                 # summary + saves outputs/<name>.stylecard.json
python scripts/analyze_cli.py a.mp4 b.mp4 --json               # print full JSON
python scripts/analyze_cli.py a.mp4 --no-cache                 # force re-analysis
python scripts/analyze_cli.py a.mp4 --mock-gemini              # FAKE Gemini labels, offline wiring test only
python scripts/make_synthetic.py                               # 5 synthetic videos with known ground truth
python scripts/check_gemini.py                                 # diagnose key/model/quota in plain English
python scripts/check_gemini.py --models                        # list models your key can call (set GEMINI_MODEL)
python -m pytest -q                                            # 37 tests, no network, no API key
```
```python
from core.pipeline import analyze_video
card = analyze_video("clip.mp4")          # -> StyleCard (pydantic); cached by SHA-256 in data/cache/
```
Without `GEMINI_API_KEY` the code-measured fields still work and Gemini-judged fields are `null` with
`confidence 0` and an entry in `meta.degraded` (nothing crashes). Degraded results are never cached.

## Pipeline
probe + normalise to a 720p H.264 CFR proxy → scenes (PySceneDetect) → audio (librosa) → color (k-means + stats; Gemini
names the grade) → motion (Farneback flow; Gemini labels shot/camera) → transitions (before/after strips → Gemini) →
text (presence pass → style burst) → whole-video overview → coupling → card. Limits: videos ≤ 60 s, friendly errors for
missing/corrupt/too-long files. Gemini calls are spaced (GEMINI_MIN_INTERVAL_S, default 5 s), retry transient errors (503 overload, 429 per-minute) with
backoff that honours the API's retry hint, fail fast on fatal ones (bad key, unknown model, daily quota), validate against
the Pydantic response schema, snap labels onto fixed vocabularies, and degrade rather than raise.
Model alias defaults to `gemini-flash-latest`; override with `GEMINI_MODEL`.

## What was verified (and what wasn't)
**Verified** on 5 synthetic videos with known ground truth (BPM, cut times, off-beat offsets, no-audio, camera shake):
scene counts exact, BPM within ~2.5%, cuts/min within 2 of truth, off-beat cuts score beat_sync 0.00, no-audio handled,
shake does not cause false cuts; Gemini client retry/validation/degradation and cache logic via unit tests (31 passing).
**NOT verified:** live Gemini calls (the build sandbox could not reach the Gemini API) — prompts, the `response_json_schema`
config, the model alias, and label quality are untested against the real service. **No real-world videos were tested.**
Run the CLI on your own references and sanity-check the cards before building on them.

## Known limitations / weak spots
- Effects are baked into pixels, so attribution (grade vs lighting, etc.) is approximate.
- Optical-flow camera hints are confused by moving subjects (false "handheld"/"pan" on steady shots); Gemini's label
  is the intended arbiter, and confidence for `camera` is 0.4 without it.
- Heuristic grade (used only when Gemini is unavailable) is uncalibrated and not very discriminative.
- Beat tracking has ~2-3% tempo error, so even perfectly on-beat cuts score ~0.9, not 1.0 (tolerance ±80 ms).
- Vignette detection can false-positive on letterboxed video. Thresholds in `color.py`/`coupling.py`/`scenes.py` are
  reasonable defaults, not tuned on labelled data.
- Gemini samples video coarsely, so fast transitions are judged from code-extracted native-fps frame strips instead.
- Cold start: first analysis in a fresh process can take ~25 s (librosa/numba warm-up); warm runs take ~6-10 s per 15-20 s clip.

---

# Stage 2 (step 1): apply engine

Borrow selected attributes from a reference's Style Card and apply them to a target clip. **No Gemini calls**: the
reference card is already analysed; the target is measured by code only.

```bash
python scripts/analyze_cli.py data/examples/opusjake.mp4      # once per reference (cached card in outputs/)
python scripts/apply_cli.py --reference outputs/opusjake.stylecard.json --target data/examples/stoic.mp4 \
    --apply pacing,audio_rhythm,transitions,color,text --out outputs/stoic_in_opusjake_style.mp4
```
Attributes: `pacing`, `audio_rhythm`, `transitions`, `color`, `text`. Options: `--strength 0..1` (color match, default 0.8),
`--zoom` (jump-cut framing change, default 1.15), `--transcript words.json` (`[{"text","start_s","end_s"}, ...]`, skips Gemini transcription; the CLI saves the Gemini transcript to `<out>.transcript.json` so you can fix it and re-render), `--caption-offset -0.1` (shift caption timing), `--words-per-caption N`, `--debug` (also writes a labelled video with a red flash at every cut, plus a before/after strip per cut). Writes the mp4 and `<out>.plan.json`.

| Attribute | What happens |
|---|---|
| color | **Rendered.** Brightness/contrast/saturation/warmth matched to the reference by a closed-loop calibration on a tiny proxy (measure, correct, repeat), then graded with FFmpeg |
| pacing | **Rendered.** Target is cut on the reference's shot-length rhythm. A continuous take gets visible *jump cuts* made by alternating framing; audio untouched, nothing removed |
| audio_rhythm | **Rendered.** Snaps those cuts to the target's beat grid (±0.3 s); falls back (and says so) if the target has no beat |
| transitions | Mix mirrors the reference's order. **Only hard cuts and zoom punch-ins are rendered**; anything else (whip pan, glitch, ...) becomes a hard cut and is logged as a fallback while the plan keeps the intended type |
| text | **Rendered as captions when a transcript is available** (automatic via Gemini audio transcription, or your own `--transcript words.json`): the reference's font (nearest open-license match; bundled in `data/fonts/`, downloaded on demand, else a system font, every step logged), size, color, case, position, background pill/outline and words-per-caption (from the measured caption rate). **Not rendered:** text animation (captions swap instantly) and varied/kinetic typography (only the dominant look). Without a transcript it stays a concrete plan |
| layout | Split-screen / picture-in-picture are **reported, not reproduced** |

Dependency warnings (e.g. transitions without pacing, pacing without audio rhythm when the reference is beat-driven, varied
kinetic typography) come from `core/bundles.py` and appear in the CLI output and the plan. The CLI also prints a
**style-match report** measured on the rendered output (color distance to the reference before/after).

## Verified vs not (Stage 2 step 1)
Verified offline on synthetic clips: 30 tests incl. real FFmpeg renders (duration, audio preserved, cut snapping, color
distance improves), font resolver, fallbacks. Visual before/after checked on a synthetic pair. **Not yet verified on your real
clips.** Known limits: the color match is statistical (it matches averages, not a look), warmth correction is capped (extreme
cool-to-warm shifts only get part-way), the scene detector may not register small-framing jump cuts, text is not rendered,
and the target must be <= 60 s.

## Captions (Stage 2 step 2)
Caption timing comes from word-level timestamps, which Gemini estimates (approximate) -- check the wording and timing, or supply
your own transcript. **The Gemini transcription call has never been run against the live API in the build sandbox** (only a fake
backend); the font download path was verified for real against GitHub, the caption rendering and layout on synthetic clips.
Fonts: 18 open-license files are bundled (`data/fonts/`, see `LICENSES.md`); other catalog fonts are fetched from the Google Fonts
GitHub repo on first use.

### Fixing caption timing locally
Gemini's word timestamps can drift for a run of words while the rest are fine, so a global `--caption-offset` is usually the wrong fix.
Use `python scripts/fix_transcript.py outputs/x.transcript.json --list` to see every word with its index and times, then shift just the
bad run: `--shift-time 11.2 13.8 -0.3` (words starting in that window, 0.3 s earlier) or `--shift-words 24 29 -0.3`. It writes
`x.transcript.fixed.json`; render with `apply_cli.py ... --apply text --transcript x.transcript.fixed.json` (no Gemini call).

---

# The app (Stage 2 step 3)

```bash
pip install -r requirements.txt
streamlit run app.py
```
Four tabs: **① Inputs** (pick a reference style card or analyse a new video with Gemini; pick or upload your clip) ·
**② Style card** (metrics, scene/beat/energy timeline, palette, closest open-license font preview, per-attribute confidence and how each
was measured) · **③ Choose what to borrow** (bundle toggles, live dependency warnings with one-click fixes, color strength, jump-cut
framing, caption timing/words-per-caption, optional transcript upload, optional debug video) · **④ Result** (before/after players,
what was done, fallback log, color-match numbers, downloads for video/plan/transcript).

Reference cards are the `outputs/*.stylecard.json` files made by `analyze_cli.py` or by the app. Overrides for tests/deployment:
`STP_CARDS_DIR`, `STP_VIDEOS_DIR`, `STP_OUT_DIR`. Without a Gemini key the app still applies styles from existing cards; analysing new
references and auto-captions need one.

Verified: 15 headless UI tests (the real script is driven through Streamlit's AppTest: card selection, warnings and one-click fixes,
apply end to end, new-reference flow with a fake Gemini) and the server boots and serves. **Not verified: how it looks in a real
browser** (layout, chart appearance, video playback), and live Gemini analysis through the UI.

### Is the transcript early or late? (`check_sync`)
`python scripts/check_sync.py clip.mp4 clip.transcript.json` compares where the transcript says speech happens against the audio
energy, and reports a global lag and confidence. Negative lag = captions appear after the words are spoken. `--fix out.json` (or
`apply_cli.py --auto-sync`, or the app's "Auto-correct caption timing" box) shifts every word by that lag, only when the estimate is
confident; the Apply plan always shows the check and suggests the flag when the transcript looks off by 0.1 s or more.
Tested on synthetic speech with known delays: the global lag is recovered to within a few ms. **Not validated on real recorded
speech.** It needs pauses between phrases to lock on and refuses when unsure. A *short run* of late words cannot be fixed
automatically (an attempt was tried and rejected: it helped about a third of the time and made one case worse), so such runs are only
listed as "suspect windows"; fix them with `scripts/fix_transcript.py`.

### Variable-frame-rate clips
Phone and downloaded clips are often variable frame rate. The caption pass now converts such sources to constant frame rate first
(real timestamps kept) so the picture is not time-warped against the audio.
