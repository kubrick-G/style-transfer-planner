# CODEX_TASKS.md: finishing the Style Transfer Planner

Read `AGENTS.md` first (project context, commands, rules, lessons). Do the tasks **one at a time, in the order given**. After
each: run the full test suite, commit, and report in the format from AGENTS.md rule 10. Stop and ask the human whenever a task
says "needs from the human".

## Recommended order and what is truly required
| # | Task | Required to send the demo? | Rough size | Needs the human? |
|---|------|---|---|---|
| T0 | Baseline: install, run tests, confirm green | yes | small | no |
| T5 | Demo content: cached reference cards + rights-cleared clips + zero-API first run | yes | medium | **yes** (clips, cards) |
| T3 | Gemini usage cap | yes (public link) | small | no |
| T4 | Public-mode hardening | yes (public link) | medium | no |
| T6 | Deployment (Streamlit Cloud, Docker/HF Spaces alternative) | yes | medium | **yes** (accounts, secrets) |
| T7 | Docs: README, WRITEUP, DEMO_SCRIPT, OUTREACH | yes | medium | **yes** (review, record demo) |
| T2 | Evaluation harness | strongly recommended (credibility) | medium | **yes** (hand labels) |
| T1 | Caption-timing investigation (open bug) | recommended | medium, uncertain | **yes** (ground-truth timings) |
| T8 | CI + cleanup | nice to have | small | no |
| T9 | Stretch features | optional | large each | review |
Rough total for the required set: 1-2 working days of agent time across 6-10 sessions, plus about half a day of human-only work.

## Human-only checklist (an agent cannot do these)
1. Rotate the Gemini API key (it was exposed in screenshots); put the new one in `.env` only. Set a spending/usage limit in Google AI Studio.
2. Choose demo clips you OWN or that are CC0/royalty-free. Do **not** put other creators' reels in the public repo or live demo.
3. Hand-label about 10 clips for the evaluation (T2) and give ground-truth word timings for a few words (T1).
4. Create the GitHub repo and the Streamlit Community Cloud (and optionally Hugging Face) account; paste secrets into their settings pages.
5. Watch every rendered output yourself. Agents cannot judge whether a video looks right.
6. Record the 60-90 s demo, review the write-up, send the outreach message.

## Running Codex on Windows (current as of Oct 2026; re-check, this changes often)
- Codex is included in ChatGPT Plus (no separate subscription); the allowance is shared with ChatGPT Work, and Plus has a rolling
  5-hour limit plus a weekly limit. Check your Codex usage dashboard. A free trial may have a smaller allowance, so do the REQUIRED tasks first.
- Install the **VS Code extension** (search "Codex" in Extensions) or the CLI (`npm install -g @openai/codex`, needs Node.js). Sign in with ChatGPT.
- Open the project folder in VS Code and commit what you have first (`git add -A ; git commit -m "baseline before codex"`) so every change is reviewable.
- Default sandbox: edits only inside the workspace, **network off**. This project needs network for `pip install`, font downloads, Gemini and git.
  Either approve each network command when asked, or enable it in `~/.codex/config.toml` (check the docs for exact keys):
  ```toml
  model = "gpt-6-sol"
  model_reasoning_effort = "medium"
  sandbox_mode = "workspace-write"
  approval_policy = "on-request"
  [sandbox_workspace_write]
  network_access = true
  [windows]
  sandbox = "elevated"
  ```
  If the native Windows sandbox misbehaves (every command asks for approval, "CreateProcessWithLogonW failed"), use WSL2 for the agent or run with approvals.
- Start every session in the project folder so `AGENTS.md` is picked up automatically. Keep prompts focused on ONE task.
- If Codex cannot read your `.env`, that is fine and safer. Do not paste the key into a prompt. For tasks that need a live key, run the command yourself and paste the output.

## Which model
Per current reporting: GPT-6 Astra is the flagship (hardest long-horizon work, costs the most of your allowance), **GPT-6 Sol is the
coding/agentic workhorse**, GPT-6 Luna is the fast, cheap tier. GPT-5.5 was retired on Oct 14, 2026.
- Default for almost every task here: **GPT-6 Sol at medium effort**; raise to high for T6 (deployment) and T1 (the timing bug).
- Use Astra only for T1 or a stubborn bug when you have allowance to spare.
- Use Luna for mechanical work (T7 docs, T8 cleanup) or when you hit a rate limit.
Confirm what your account shows in `/model` (the picker lists what you can use) and treat names above as a snapshot.

## Kickoff prompt (paste once per new session; replace TASK)
```
Read AGENTS.md and CODEX_TASKS.md fully before doing anything. Then do ONLY task TASK from CODEX_TASKS.md.
Follow the engineering rules in AGENTS.md (offline tests, honest labelling, fail-soft, bump SCHEMA_VERSION if the card changes,
never touch or print .env or API keys, quote Windows paths). Do not start any other task. When finished, run the full test suite and
report in the AGENTS.md format (Changed / Tests / Verified by running / NOT verified / Follow-ups). If something needs me, stop and ask.
```

---
## T0: Baseline
**Goal:** prove the repo runs on this machine before changing anything.
**Steps:** create/activate the venv; `pip install -r requirements.txt`; confirm `ffmpeg -version` (>= 5.1); `python -m pytest -q`; `python -m streamlit run app.py` and confirm the page loads.
**Acceptance:** 180 tests pass (or every failure is explained and fixed WITHOUT changing behaviour); app boots. If Python 3.14 package issues appear, pin working versions in `requirements.txt` and say which.
**Prompt:** kickoff + "TASK = T0. Also list anything environment-specific you had to change."

## T5: Demo content (do this before deployment)
**Goal:** the public demo must work with **zero Gemini calls** on first load, using only content we have the right to ship.
**Needs from the human:** 2-3 reference reels they can legally include (their own, or CC0), 2 short target clips (<= 20 s, own footage), and a sample transcript JSON for one target. Reference cards are made by running `scripts/analyze_cli.py` on the references (uses their key); the agent never needs the key.
**Steps:**
1. Add `data/demo/cards/` (committed `*.stylecard.json`), `data/demo/clips/` (small mp4s, commit only if rights-cleared and each < 5 MB), `data/demo/transcripts/`.
2. Make `core/ui_helpers.cards_dir()`/`videos_dir()` logic list cards from BOTH `outputs/` and `data/demo/cards/` (demo cards shown with a "demo" tag; user cards override on name clash). Keep the `STP_*` overrides working.
3. First-run experience: with an empty `outputs/` and no API key, a visitor can pick a demo reference + demo clip, apply pacing/color/transitions, and (using a bundled demo transcript) captions.
4. `scripts/export_demo_assets.py`: copies chosen cards/clips/transcripts into `data/demo/` and strips nothing else; refuses files > 5 MB without `--force`.
5. If no rights-cleared clips exist yet, generate synthetic ones with `scripts/make_synthetic.py` so the app is demonstrable, and say clearly in the README that they are placeholders.
**Acceptance:** AppTest passes for the zero-key path (select demo card + clip, apply, result exists); tests prove user cards override demo cards; no file in `data/demo/` > 5 MB; `git ls-files data/demo` contains no video the human has not approved.
**Prompt:** kickoff + "TASK = T5. Ask me for the clips and cards; until I provide them use synthetic placeholders."

## T3: Gemini usage cap
**Goal:** a public link must not burn the owner's quota.
**Steps:** add `core/usage_cap.py`: `UsageCap(path, daily_limit)` with a JSON counter keyed by UTC date, safe for concurrent reads/writes (file lock or atomic replace), `remaining()`, `charge(n)`, limit from `STP_DAILY_GEMINI_LIMIT` (default 40). Add `CappedGeminiClient` (subclass or wrapper of `GeminiClient`) that charges one unit per real request ATTEMPT (retries count) and, when exhausted, behaves like an unavailable client with a clear reason. Use it in `app.py` and `apply_cli.py`/`analyze_cli.py`. The app sidebar shows "Gemini requests left today: N" and explains what is disabled when it reaches 0 (analysis, captions) versus what still works (everything else).
**Acceptance:** unit tests for rollover at UTC midnight (inject a clock), concurrency, limit from env, blank/invalid env ignored; AppTest shows the counter and disabled state; exhausted cap never calls the backend.
**Prompt:** kickoff + "TASK = T3."

## T4: Public-mode hardening
**Goal:** safe, predictable behaviour on a shared free host.
**Steps:** `STP_PUBLIC_MODE=1` enables: clip length limit `STP_MAX_CLIP_S` (default 30) and upload size limit (default 50 MB) enforced BEFORE heavy work with friendly messages; "Analyze a new reference" disabled for visitors (demo references only) unless `STP_ALLOW_ANALYSIS=1`; uploads and renders older than 1 hour deleted at app start and after each run; one heavy job at a time (a semaphore with a "busy, try again in a minute" message); a visible privacy note ("clips are processed on the server and deleted within an hour; do not upload private footage"); timeouts around ffmpeg calls (already in `run()`), and a total apply time budget with a clear failure message.
**Acceptance:** AppTest covers each limit; unit tests for the cleanup of old files (inject a clock); public mode off keeps current behaviour exactly; no upload is ever written outside `STP_OUT_DIR`.
**Prompt:** kickoff + "TASK = T4."

## T6: Deployment
**Goal:** a public URL that works for a stranger in an incognito window, plus a Docker fallback.
**Needs from the human:** GitHub repo, Streamlit Community Cloud account, secrets pasted into the app settings.
**Steps:**
1. Split `requirements.txt` into runtime and `requirements-dev.txt` (pytest). Pin versions that were verified to install together (record the Python version used; prefer 3.12 for hosting).
2. Verify `packages.txt` (ffmpeg, libgl1, libsm6, libxext6) suffices on the host image; add anything missing.
3. Secrets via `st.secrets`/environment: `GEMINI_API_KEY`, `GEMINI_MODEL`, `GEMINI_MIN_INTERVAL_S`, `STP_PUBLIC_MODE=1`, `STP_DAILY_GEMINI_LIMIT`. Make the app read `st.secrets` as a fallback to env without ever printing them.
4. `Dockerfile` (python:3.12-slim + ffmpeg + libgl1 + libglib2.0-0; `streamlit run app.py --server.port 7860 --server.address 0.0.0.0`) and the Hugging Face Spaces front matter (`sdk: docker`, `app_port: 7860`) as an alternative host.
5. `scripts/smoke_test.py URL_OR_LOCAL`: checks `/_stcore/health`, then runs `apply_style` on a bundled demo pair headless and prints timings.
6. Measure on the real host (human runs it): cold start, one full apply on a 20 s clip, peak memory. Write the numbers into the README. Free tiers have small memory and CPU; check the host's CURRENT limits in its docs and lower `STP_MAX_CLIP_S` if needed.
**Acceptance:** `docker build` and local container run pass the smoke test; deployment guide in `docs/DEPLOY.md` with exact click-by-click steps; no secret in git (`git grep -n "AIza"` is empty); the live URL loads with an empty cache and no key.
**Prompt:** kickoff (high effort) + "TASK = T6. Prepare everything in the repo; tell me exactly what I must do in the Streamlit/GitHub/HF web pages."

## T7: Documentation and outreach
**Goal:** what the recipient reads. Honest, specific, short.
**Steps:** rewrite `README.md` (what it is, the Cardboard problem it addresses, a Mermaid architecture diagram, how each attribute is measured (code vs Gemini), setup, deploy link, **Verified vs Not verified**, known limitations, what comes next with a real editing engine); `WRITEUP.md` (one page: problem, approach, key decisions (code-for-objective / AI-for-subjective; style card with provenance and confidence; closed-loop color on the real pipeline; honest fallback log; refusing unreliable corrections), what worked, what failed and was removed, evaluation results from T2); `DEMO_SCRIPT.md` (60-90 s, exact clicks); `OUTREACH.md` (4-6 sentences, link first).
**Rules:** every number comes from a measurement in the repo (cite the command); no claim of accuracy that was not evaluated; mention the open caption-timing issue if T1 is unresolved. Map to Cardboard's list: #6 Style references (core), #7 Motion graphics (caption animation is a stated gap), #8 Memory (style cards as portable preferences), #9 Verification (style-match report, sync check).
**Acceptance:** all commands in the docs run as written (agent executes them); a reader can reproduce the demo; no broken links.
**Prompt:** kickoff (Luna is fine) + "TASK = T7. Draft everything; I will edit the voice."

## T2: Evaluation harness
**Goal:** measure how often Gemini's labels agree with a human's, and how stable they are between runs.
**Needs from the human:** `eval/labels.json` for ~10 clips (template provided by the agent), plus the clips and their cards.
**Steps:** `eval/labels_template.json` with, per clip: layout, grade, caption position band, text animation, font category, per-scene shot types, per-cut transition types, scene count and cut times (seconds), BPM. `eval/run_eval.py --labels eval/labels.json --cards-dir outputs` prints per-attribute accuracy with n, confusion summaries for transitions and grade, mean absolute error for cut times and BPM, and writes `eval/results.md`. `--live --repeat 2` re-analyses each clip `--no-cache` twice and reports run-to-run agreement per attribute; it MUST have `--max-calls` and print the estimated request count before starting (a 10-clip, 2-run evaluation is ~200 requests). With no labels it runs consistency-only and says so.
**Acceptance:** tests with fake cards/labels; results file marks placeholder data clearly; small-sample caveat printed ("n=10: indicative only").
**Prompt:** kickoff + "TASK = T2. Do not run --live yourself; give me the exact command and I will run it."

## T1: Caption-timing investigation (open bug, time-boxed)
**Context:** see AGENTS.md "Open bug". **Do not** reintroduce local auto-correction.
**Needs from the human:** for `lv_0_20231212164716.mp4`, the time they hear 5 specific words (e.g., "dropshipping", "money", "business", "treated", "2018") to the nearest ~0.1 s, in `eval/sync_truth/lv_0.json` as `[{"text","true_start_s"}]`.
**Steps:** (1) `scripts/sync_truth_report.py` compares the saved Gemini transcript to the truth and classifies the error: constant offset, linear drift, or local. (2) Fix by class: constant → document `--caption-offset` with the measured value; drift → add `--caption-anchors` (two or more word/time pairs, linear fit) ; local → document `fix_transcript.py`. (3) Optionally add `--align whisper` using `faster-whisper` as an OPTIONAL dependency (`requirements-align.txt`, lazy import, CPU int8 `base` model, degrade with a clear message if absent), and ship it only if its errors on the truth set are lower than Gemini's.
**Time-box:** if no cause is found after the truth report, write up the numbers and stop; do not guess.
**Acceptance:** report with per-word errors; every code change validated on that clip's truth set; tests for the report, the anchor fit, and the optional-import fallback.
**Prompt:** kickoff (Astra if allowance allows, else Sol high) + "TASK = T1. Wait for my truth file before coding."

## T8: CI and cleanup
**Steps:** `.github/workflows/tests.yml` (ubuntu-latest, apt install ffmpeg libgl1, Python 3.12, `pip install -r requirements-dev.txt`, `pytest -q`, pip cache); rename `tests/test_patch3.py`, `test_patch5.py`, `test_patch6.py` into meaningful modules without losing tests; remove dead code only if unused (prove with grep and tests); add `ruff` config and fix only safe lint; README badge.
**Acceptance:** CI green on a PR; test count unchanged or higher.

## T9: Stretch (each is its own task; only after T0-T8; the human must review visually)
a. Render caption **pop/scale** and fade entrance animations (currently a logged fallback).
b. **Crossfade** transitions via FFmpeg `xfade` at planned cuts; a simple whip-pan approximation.
c. **Layout transfer** for split-screen / picture-in-picture references.
d. **Per-scene typography** for kinetic-text references (one dominant style per scene).
e. Faster caption timing from forced alignment (see T1).
Each needs: tests with synthetic ground truth, an honest fallback entry for whatever is still unsupported, README update, and a human visual check.

## After every task, the human should
`git diff --stat` and skim it; run `python -m pytest -q`; run the app and try the changed feature; run `git grep -n "AIza"` (must be empty);
and paste anything surprising back into the chat where this project was built so it can be reviewed.
