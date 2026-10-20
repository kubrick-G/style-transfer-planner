# Demo Script: Style Transfer Planner (60-90s)

A crisp, high-signal demonstration walkthrough designed for the Cardboard engineering team.

---

## Pre-Demo Checklist
1. Start the app: `python -m streamlit run app.py`
2. Open browser to `http://localhost:8501`.
3. Ensure window is at 1080p recording resolution with audio enabled.

---

## Timeline & Narration

### [00:00 - 00:15] Introduction & The Problem
- **Visual**: Show Streamlit app landing page with Tab ① (Inputs).
- **Narration**:  
  *"Hey Cardboard team! When creators want to adopt the editing style of a reference reel, generative video models hallucinate and wipe out the footage. Here is **Style Transfer Planner**, an agentic, structured approach to your Hard Problem #6."*

### [00:15 - 00:35] Step 1: The Style Card
- **Action**: Click on Tab ② **Style card**.
- **Visual**: Scroll past the detected scenes metric, BPM, timeline chart (scenes colored by energy and beat grid), and color palette.
- **Narration**:  
  *"Instead of an end-to-end black box, we decompose the reference video into a machine-readable Style Card. Crucially, code measures anything objective—like cuts, BPM, and color statistics—while Gemini only handles subjective nuances like typography aesthetic and scene roles."*

### [00:35 - 00:55] Step 2: Choosing What to Borrow
- **Action**: Click on Tab ③ **Choose what to borrow**.
- **Action**: Toggle "Pacing" off, then observe the automatic dependency warning: *"Transitions are placed on the cuts"*. Click the one-click **"Add Pacing"** button to resolve it.
- **Narration**:  
  *"Styles live in combinations. Attributes are organized into bundles with data-driven dependency warnings so creators don't inadvertently create jarring edits."*

### [00:55 - 01:20] Step 3: Applying Style & Reviewing Fallbacks
- **Action**: Select target clip `demo_target_speech.mp4 (demo)`. Select demo transcript `demo_target_speech.transcript.json`.
- **Action**: Click **"Apply style"**.
- **Visual**: Status expands showing closed-loop color calibration, beat snapping, and caption burn-in completing in under 10 seconds.
- **Action**: Switch to Tab ④ **Result**.
- **Visual**: Play the side-by-side Before / After comparison. Scroll down to highlight the **Fallback Log** and **Style Match Report**.
- **Narration**:  
  *"In Tab 4, we see our continuous take converted into dynamic beat-synced jump cuts with calibrated color and matched typography. Most importantly, nothing fails silently: the Fallback Log explicitly documents what couldn't be rendered and what open-source asset was used instead."*

### [01:20 - 01:30] Wrap-up
- **Visual**: Show GitHub repo and test suite badge.
- **Narration**:  
  *"180 offline tests, closed-loop calibration, and zero-key demo readiness. I'd love to bring this structured editing intelligence to Cardboard!"*
