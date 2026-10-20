"""All Gemini prompts, as named constants. Batch prompts carry a machine-readable N_ITEMS token."""

SYSTEM_HINT = (
    "You are a precise video-editing analyst. Answer ONLY with JSON matching the requested schema. "
    "Use ONLY the allowed vocabulary values where a list is given. If unsure, pick the closest value "
    "and lower your confidence (0..1)."
)

GRADE = (
    "The images are 3 frames sampled evenly across one short vertical video.\n"
    "Statistics measured by code (treat as ground truth; do NOT contradict them): {stats}\n"
    "1) Name the overall color grade with exactly one of: {grades}.\n"
    "2) Say whether visible film grain / noise texture is an intentional stylistic element (grain: true/false).\n"
    "Return JSON: grade, grain, confidence."
)

SCENE_LABELS = (
    "Each image is the middle frame of one scene, in order. N_ITEMS={n}\n"
    "For every scene give: shot_type (one of: {shots}) and camera_motion (one of: {moves}).\n"
    "camera_motion describes the PHYSICAL camera filming a person or subject. Ignore on-screen graphics, text and "
    "screen recordings: for screen-recording shots use 'static'. A person filmed without any movement is 'static'. "
    "If the frame has panels (e.g. a screen recording above a facecam), judge the camera footage of the person. "
    "Use 'other' only as a last resort.\n"
    "Use 'motion-graphic' for full-screen designed cards/typography/sticker slides that are not filmed footage "
    "or screen recordings.\n"
    "A camera-motion HINT measured by optical flow is given per scene; use it unless the image clearly "
    "contradicts it (flow is confused by moving subjects and by on-screen panels changing).\n"
    "Return JSON: scenes=[{{index, shot_type, camera_motion}}], confidence."
)

TRANSITIONS = (
    "Each image is a 2-row strip around one cut: top row = last frames BEFORE the cut, bottom row = first "
    "frames AFTER the cut (time flows left to right). N_ITEMS={n}\n"
    "Classify the transition at each cut as exactly one of: {types}.\n"
    "Guidance: whip pan = heavy directional motion blur sweeping into the next shot; zoom punch-in = sudden "
    "scale-up around the cut; crossfade = blended overlap of both shots; flash = bright/white frame at the cut; "
    "glitch = digital distortion/RGB split; slide/push = one shot pushes the other off-screen; hard cut = clean "
    "instantaneous change with no effect. An 'abrupt' hint measured by code is given per cut (abrupt=true means "
    "the pixel change happens within a single frame).\n"
    "Return JSON: transitions=[{{index, type}}], confidence."
)

TEXT_PRESENCE = (
    "Each image is a frame sampled about once per second, labelled with its index. N_ITEMS={n}\n"
    "For every frame say whether it contains on-screen TEXT ADDED IN EDITING: captions/subtitles, titles, "
    "lower-thirds, overlay graphics. Do NOT count text that is part of a screen recording or the filmed scene "
    "(terminal windows, app interfaces, slides, websites, signs, clothing). If yes, give a TIGHT normalised "
    "bounding box [x0,y0,x1,y1] in 0..1 around that caption/overlay text only.\n"
    "Return JSON: frames=[{{index, has_text, bbox}}]."
)

TEXT_STYLE = (
    "The first images are a short burst (about 0.3 s apart, in order) around the moment on-screen text first "
    "appears; the remaining images are other frames showing the text. Describe the text STYLE (not the words).\n"
    "Return JSON with: font_category (one of {cats}), weight (one of {weights}), width (one of {widths}), "
    "geometry (one of {geoms}), mood (2-3 short tags), case (one of {cases}), position (where the "
    "captions/overlay text sit, one of {positions}), color (dominant text color as #rrggbb), "
    "outline_or_shadow (one of {outlines}), chunking - how much text is on screen at once (one of {chunks}), "
    "animation - how the text enters, judged from the burst (one of {animations}), confidence.\n"
    "Only describe captions/overlay text added in editing, NOT text inside screen recordings or the filmed scene.\n"
    "variety: 'consistent' if one typographic treatment is used throughout; 'varied' if fonts, colors, sizes or "
    "positions change a lot between moments (kinetic typography). When 'varied', give the attributes of the "
    "DOMINANT treatment. description: ONE sentence (max 25 words) summarising the overall text treatment.\n"
    "Font-category check: choose 'mono' when every letter has the same width (e.g. a narrow 'I' still looks wide, "
    "with slab-like serifs, typewriter/terminal/code look); 'sans' only for proportional sans-serif. Also report "
    "outline_or_shadow = 'background-box' when the text sits on a filled box or pill.\n"
    "animation 'none' = text simply appears/disappears with no entrance effect. Captions that merely replace "
    "each other word by word are still 'none'; use 'word-by-word' only if words visibly animate in (pop, slide, "
    "highlight).\n"
    "Do NOT name a specific font."
)

OVERVIEW = (
    "This is a short vertical video. Give: mood (a few words, e.g. 'energetic, cinematic'), genre_guess for the "
    "music or overall vibe (a few words), and a sanity check of its structure: does it have a clear hook in the "
    "first seconds, a body, and an ending? structure_ok (bool) + structure_notes (one sentence). Also give layout "
    "(one of: {layouts}): how the frame is composed; split-screen = two or more panels such as a screen recording "
    "above a facecam; picture-in-picture = a small inset over full-frame footage. Return JSON: "
    "mood, genre_guess, layout, structure_ok, structure_notes, confidence."
)

RETRY_REMINDER = "Your previous reply was not valid JSON for the required schema. Reply with ONLY valid JSON."

TRANSCRIPT = (
    "Transcribe the SPEECH in this audio (about {duration} seconds) with word-level timestamps in seconds from the "
    "start of the audio. One entry per spoken word, in order, with the word exactly as spoken (keep the language of "
    "the speech; do not translate). Skip music and non-speech sounds. Do not invent words you cannot hear. "
    "Return JSON: words=[{{text, start_s, end_s}}]."
)
