const fs = require('node:fs');
const path = 'test_l1.ipynb';
const nb = JSON.parse(fs.readFileSync(path, 'utf8'));
const cell = tag => nb.cells.find(c => c.metadata.tags?.includes(tag));
const read = tag => cell(tag).source.join('');
const set = (tag, text) => { cell(tag).source = text.match(/.*(?:\n|$)/g).filter(Boolean); };
const replace = (text, old, value) => {
  if (!text.includes(old)) throw Error('Missing replacement: ' + old);
  return text.replace(old, value);
};
set('setup', replace(read('setup'),
  'HAND_CONTROL = False  # Colab Level 3 only; local mode currently uses XYZ sliders.',
  'HAND_CONTROL = True  # Level 3. G toggles local hand/manual control.\nHAND_MODEL_PATH = "models/hand_landmarker.task"  # Local asset, prepared below.\nSHOW_HAND_LANDMARKS = True  # Output-only dots; never enter depth inference.'));
set('install', '# Optional dependency setup. Keep your CUDA-compatible torch/torchvision build.\n' +
  '# MediaPipe needs contrib OpenCV; install only one cv2 provider. Restart kernel afterwards.\n' +
  '%pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python-headless\n' +
  '%pip install torch torchvision "numpy>=1.24,<2" matplotlib pillow "opencv-contrib-python>=4.8,<4.12" "transformers>=4.40,<5" "mediapipe==0.10.32"\n');
const setup = `# Run once before the live cell for local Level 3. No camera is opened here.
# The versioned Google model is cached in the notebook working directory.
def prepare_hand_model(path):
    from pathlib import Path
    from urllib.request import urlopen
    import os
    import tempfile
    target = Path(path)
    if target.is_file():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    url = ('https://storage.googleapis.com/mediapipe-models/'
           'hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task')
    temporary = None
    try:
        with urlopen(url, timeout=30) as response, tempfile.NamedTemporaryFile(
                dir=target.parent, suffix='.part', delete=False) as stream:
            temporary = Path(stream.name)
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                stream.write(chunk)
        if temporary.stat().st_size < 1024:
            raise RuntimeError('Hand model download was incomplete')
        os.replace(temporary, target)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return target

if LIVE_BACKEND == 'local' and HAND_CONTROL:
    try:
        print('Local hand model:', prepare_hand_model(HAND_MODEL_PATH))
    except Exception as error:
        print(f'Hand model unavailable: {error}. Live sliders remain available; rerun to retry.')
`;
nb.cells.splice(nb.cells.indexOf(cell('live')), 0, {
  cell_type: 'code', execution_count: null, metadata: {tags: ['hand-model']}, outputs: [],
  source: setup.match(/.*(?:\n|$)/g).filter(Boolean)
});
let live = read('live');
live = replace(live, 'class LatestLocalCamera:', fs.readFileSync('.phase3-local-helpers.py', 'utf8') + '\n\nclass LatestLocalCamera:');
live = replace(live, 'def reset_local_light():\n    values = default_light()',
  'def reset_local_light():\n    values = default_light()\n    session = globals().get("_local_gesture")\n    if session is not None:\n        session.light = dict(values)\n        session.calibrate()');
live = replace(live, 'global _local_camera, _local_window', 'global _local_camera, _local_window, _local_gesture');
live = replace(live, '    old = globals().get("_local_camera")\n    if old is not None:\n        old.close()',
  '    if globals().get("_local_camera") is not None or globals().get("_local_gesture") is not None:\n        stop_webcam()');
live = replace(live, '_local_window = "AI Vision - relighting (Q/Esc: stop, R: reset)"',
  '_local_gesture = LocalGestureSession()\n    _local_window = "AI Vision - Level 3 (G: hand/manual, C: calibrate, R: reset, Q: stop)"');
live = replace(live, '        reset_local_light()\n        return True',
  '        reset_local_light()\n        if globals().get("HAND_CONTROL", False):\n            _local_gesture.enable()\n        return True');
live = replace(live, '    return rgb, light, hand\n\n\ndef show_local_frame',
  '    light, tracking = _local_gesture.update(rgb, light)\n    hand.update(tracking)\n    hand["age_ms"] += tracking["hand_ms"]\n    sync_local_xyz(light)\n    return rgb, light, hand\n\n\ndef show_local_frame');
live = replace(live, '    if key == ord("r"):\n        reset_local_light()',
  '    if key == ord("r"):\n        reset_local_light()\n    elif key == ord("g"):\n        _local_gesture.toggle()\n    elif key == ord("c"):\n        _local_gesture.calibrate()');
live = replace(live, '    global _local_camera\n    camera = globals().get("_local_camera")',
  '    global _local_camera, _local_gesture\n    camera = globals().get("_local_camera")');
live = replace(live, '    finally:\n        # Retain an unclosed camera reference',
  '    finally:\n        try:\n            session = globals().get("_local_gesture")\n            if session is not None:\n                session.close()\n                _local_gesture = None\n        finally:\n            close_local_window(camera)\n\n\ndef close_local_window(camera):\n    global _local_camera\n    # Retain an unclosed camera reference');
// The old finally body moves to the cleanup helper.
live = replace(live, '    # Retain an unclosed camera reference so a rerun retries its cleanup.\n        if camera is None or not camera.thread.is_alive():\n            _local_camera = None\n        try:\n            cv2.destroyWindow(globals().get("_local_window", "AI Vision"))\n            cv2.waitKey(1)\n        except cv2.error:\n            pass',
  '    # Retain an unclosed camera reference so a rerun retries its cleanup.\n    if camera is None or not camera.thread.is_alive():\n        _local_camera = None\n    try:\n        cv2.destroyWindow(globals().get("_local_window", "AI Vision"))\n        cv2.waitKey(1)\n    except cv2.error:\n        pass');
live = replace(live, '            frame_rgb, light, hand = packet',
  '            frame_rgb, light, hand = packet\n            hand_ms = hand.get("hand_ms", 0.0)\n            capture_ms = max(0.0, capture_ms - hand_ms)');
live = replace(live, '                picture = picture.copy()',
  '                picture = picture.copy()\n                annotate_local_hand(picture, hand, h, w)');
live = replace(live, 'stage_ms.update(capture=capture_ms, download_wait=download_ms,',
  'stage_ms.update(capture=capture_ms, hand=hand_ms, download_wait=download_ms,');
live = replace(live, 'keys = ("capture", "preprocess"', 'keys = ("capture", "hand", "preprocess"');
live = replace(live, 'Use the OpenCV window sliders for XYZ, power and specular. Press **Q/Esc** or close the window to stop.',
  'Move one palm for XYZ. **G**: hand/manual; **C**: recalibrate Z; **R**: reset; **Q/Esc**: stop. Power/specular sliders stay active.');
set('live', live);
set('intro', read('intro').replace('local Level 2 uses OpenCV sliders; browser hand tracking requires Google Colab.',
  'local Level 3 uses MediaPipe hand tracking, with OpenCV sliders as fallback.').replace(
  'Level 3 adds browser-side hand tracking', 'Level 3 adds local and optional browser hand tracking'));
set('live-notes', read('live-notes').replace('## Local live camera — Levels 01–02',
  '## Local live camera — Levels 01–03').replace('Use `opencv-python`, not its headless build.',
  'Install `requirements-notebook.txt` (GUI `opencv-contrib-python`) and restart the kernel.\nIf migrating from Level 2, run the optional install cell to remove other cv2 providers.').replace(
  'Local gesture tracking is not implemented here; Level 2 uses the window sliders.',
  `
**Local hand control is enabled by default.** Run the hand-model setup cell once
to download Google's versioned Hand Landmarker asset. Subsequent runs use the
cached file. Missing packages/model or tracking errors leave sliders available.

- Show one open, front-facing palm for three valid processed frames to acquire.
- Move right/down in the **unmirrored output** to increase light X/Y.
- Move closer to decrease Z; move farther away to increase it. First acquisition
  calibrates relative Z. Press **C** to recalibrate at a comfortable distance.
- Press **G** for hand/manual mode (or retry after an error), **R** to reset
  light and calibration, **Q/Esc** to stop. Focus the OpenCV window for keys.
- XYZ sliders display the current light and are overridden in gesture mode.
  Switch to manual to edit XYZ. Power and specular always remain adjustable.
- Missing, clipped, or tiny palms hold the last light. Reacquisition needs three
  valid frames. Resolution changes reset calibration and the VIDEO tracker.
  Use one hand; identity is not locked when multiple hands enter the frame.

The local mapper uses the same formulas and 120 ms smoothing described below
for Colab. Palm rotation can mimic Z motion; this is a relative depth gesture,
not metric hand position. Green landmark dots are drawn only on the final
output; set SHOW_HAND_LANDMARKS=False for clean relighting.

Local CPU MediaPipe VIDEO detection runs once on each processed camera frame,
before depth inference. This keeps controls, landmarks and geometry aligned.
The capture thread continues reading during processing; there is no inference
queue. Hand inference currently adds serial work, reported as the **hand** stage
and included in completed-loop FPS. Live responsiveness and FPS must be measured
on the demo machine; the browser's independent tracker rate does not apply locally.

API reference: [MediaPipe Hand Landmarker for Python](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/python).
`).replace('1. **Hand control** is off by default for faster Level 2 relighting. Enable the checkbox for Level 3.',
  '1. **Hand control** follows HAND_CONTROL in setup (enabled by default). Disable it for Level 2.').replace(
  'relighting-only output, status once per second, and hand tracking off.',
  'relighting-only output and status once per second. Set HAND_CONTROL=False to disable tracking.'));
for (const c of nb.cells) if (c.cell_type === 'code') { c.outputs = []; c.execution_count = null; }
fs.writeFileSync(path, JSON.stringify(nb, null, 1) + '\n');
