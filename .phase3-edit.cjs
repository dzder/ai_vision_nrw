const fs = require('node:fs');
const path = 'test_l1.ipynb';
const nb = JSON.parse(fs.readFileSync(path, 'utf8'));
const cell = tag => nb.cells.find(c => c.metadata.tags?.includes(tag));
const set = (c, s) => { c.source = s.match(/.*(?:\n|$)/g).filter(Boolean); };
const browser = fs.readFileSync('.phase3-browser.js', 'utf8');
let live = cell('live').source.join('');
live = live.replace(/display\(Javascript\(r"""[\s\S]*?"""\)\)/,
    'display(Javascript(r"""\n' + browser + '\n    """))');
live = live.replace('data["light"]\n', 'data["light"], data["hand"]\n');
live = live.replace('frame_rgb, light = packet', 'frame_rgb, light, hand = packet');
live = live.replace('"Level 2: relit RGB"', '"Level 3: hand-controlled light"');
live = live.replace('"Use the sliders to move the light; **Stop camera** or interrupt to finish."',
    'f"**Hand control:** {hand[\'mode\']} | {hand[\'status\']}  \\n"\n' +
    '                f"**Capture age at transfer:** {hand[\'age_ms\']:.0f} ms (excludes inference/display latency)  \\n"\n' +
    '                "Move one palm for XYZ; use **Calibrate hand Z** to set a new reference. "\n' +
    '                "Uncheck **Hand control** for sliders; **Stop camera** or interrupt to finish."');
set(cell('live'), live);
set(cell('intro'), cell('intro').source.join('').replace('# Levels 01–02 — perspective geometry and dynamic relighting',
    '# Levels 01–03 — geometry, relighting, and spatial gesture control').replace(
    'Level 2 adds a virtual point light', 'Level 3 adds browser-side hand tracking for smooth XYZ light control, with\ncalibration, tracking feedback, and manual fallback. See the live instructions.\n\nLevel 2 adds a virtual point light'));
set(cell('live-notes'), `## Level 03 — spatial gesture control (live Colab experiment)

Run the cells above, skip the optional photo if desired, then run the live cell
and allow camera access. Keep this output visible and the browser tab active.
The first run downloads a pinned MediaPipe Tasks Vision JS/WASM runtime
(version 0.10.22) and the Hand Landmarker model in the browser. Python needs no
additional package. Manual controls work while loading and if loading fails.

1. **Hand control** is enabled by default. Show **one open palm facing the
   camera** for three tracking updates. Green dots show detected landmarks.
2. Move the palm right/left and up/down in the **unmirrored preview** to move
   light X/Y. Preview, depth, and relighting share the same orientation.
3. Move the palm **closer to the camera** to decrease light Z; move it away to
   increase Z. The first steady palm defines the current Z reference. Use
   **Calibrate hand Z** at a comfortable distance to recalibrate (also after
   changing hands). Keep the palm facing the camera: rotation changes its
   apparent size and can mimic depth motion.
4. Missing, clipped, or very small palms **hold the last light position**.
   Reacquisition requires three valid updates and resumes with smoothing.
   Use one hand; identity is not locked when multiple hands enter the frame.
5. Power and Specular remain adjustable. Uncheck **Hand control** to enable
   manual XYZ sliders. **Reset light** restores defaults and resets calibration.
   **Stop camera** or interrupt the cell to release the camera and tracker.

Mapping: palm center = mean of landmarks 0/5/9/13/17; X = 6(u−0.5),
Y = 4(v−0.5). Palm span is the 5-to-17 knuckle distance with both axes
normalized by image width. Z = reference_Z − 1.5 ln(span/reference_span),
clamped to [−2, 0.7]; X/Y are clamped to [−3, 3]/[−2, 2]. A time-based
EMA with 120 ms time constant smooths the light; a capped time step avoids
a jump after a dropout. This is a **relative gesture depth proxy**, not meters
or camera-relative depth from MediaPipe's wrist-relative landmark Z.

The browser tracks at **up to 20 updates/s**, independently of Python depth
inference. Each synchronous WASM detection can block the browser UI; this is
a rate cap, not a measured speed. There is no queue: each Python capture takes
the latest raw tracked canvas plus a copy of its light/status. Green overlay
dots stay out of model input. The 2×2 RGB/Z/normals/relit display remains
frame-aligned. Capture age reports stale browser frames, while the existing
completed-cycle rate includes Colab transfers and depth/render work. Neither
is sensor-to-screen latency. Background tabs may throttle tracking.

Live Colab camera behavior and throughput must be measured on the demo machine;
this notebook does not yet implement Level 4 cast shadows.

API reference: [MediaPipe Hand Landmarker for Web](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/web_js).
`);
for (const c of nb.cells) if (c.cell_type === 'code') { c.outputs = []; c.execution_count = null; }
fs.writeFileSync(path, JSON.stringify(nb, null, 1) + '\n');
