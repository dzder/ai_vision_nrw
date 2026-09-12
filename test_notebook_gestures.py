"""Exercise the notebook's local Level 3 code without a camera or GUI.

--model-smoke also uses the installed MediaPipe and cached hand model.
--download-model prepares Google's versioned model using the notebook helper.
"""
import argparse
import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from test_notebook_performance import load_helpers


def palm(x=0.5, y=0.5, span=0.2):
    points = [SimpleNamespace(x=x, y=y, z=0.0) for _ in range(21)]
    points[5].x, points[17].x = x - span / 2, x + span / 2
    return points


def check_mapping(ns):
    control = ns['PalmLightControl']()
    light = ns['default_light']()
    original = dict(light)
    for t in (0, 0.05):
        assert 'Acquiring' in control.update(palm(), 640, 480, t, light)
        assert light == original
    assert control.update(palm(), 640, 480, 0.1, light).startswith('Tracking palm')
    assert light['x'] > original['x'] and light['y'] > original['y']
    assert light['z'] == original['z']
    before = dict(light)
    control.update(palm(0.7, 0.7, 0.4), 640, 480, 0.15, light)
    assert light['x'] > before['x'] and light['y'] > before['y']
    assert -2 < light['z'] < before['z']  # Closer palm -> smaller Z, smoothed.
    control.update(palm(span=0.1), 640, 480, 0.2, light)
    assert light['z'] > -0.36
    assert light['power'] == original['power'] and light['specular'] == original['specular']

    invalid = [None, [], palm(span=0.01), palm(x=-0.1), palm(x=float('nan'))]
    for points in invalid:
        before = dict(light)
        assert 'holding' in control.update(points, 640, 480, 0.25, light)
        assert light == before
        for t in (0.30, 0.35):
            assert 'Acquiring' in control.update(palm(), 640, 480, t, light)
            assert light == before
        control.update(palm(), 640, 480, 0.40, light)

    before = dict(light)
    assert 'Acquiring' in control.update(palm(), 640, 480, 2.0, light)
    assert light == before  # Stall doesn't jump the light.
    assert 'Acquiring' in control.update(palm(), 320, 240, 2.05, light)
    assert control.reference is None  # New shape needs its own reference.
    control.reset()
    light['z'] = -0.8
    for t in (3.0, 3.05, 3.1):
        control.update(palm(span=0.35), 320, 240, t, light)
    assert light['z'] == -0.8
    for i in range(100):
        control.update(palm(0.5, 0.5, 1), 320, 240, 3.15 + i * .05, light)
    assert -2 <= light['z'] <= .7 and -3 <= light['x'] <= 3 and -2 <= light['y'] <= 2

    # Equal normalized physical palm spans at different aspect ratios -> same Z.
    results = []
    for h in (480, 360):
        c, value = ns['PalmLightControl'](), ns['default_light']()
        for i in range(4):
            points = palm(span=0.2)
            points[5].y, points[17].y = .5 - .03 * 640 / h, .5 + .03 * 640 / h
            c.update(points, 640, h, i * .05, value)
        results.append(c.reference[0])
    np.testing.assert_allclose(results[0], results[1])


def check_session(ns):
    real_tracker = ns['LocalHandTracker']
    created = []

    class Tracker:
        fail = False

        def __init__(self, path):
            self.closed = False
            self.frames = []
            created.append(self)

        def detect(self, rgb, now):
            self.frames.append(rgb.copy())
            if self.fail:
                raise RuntimeError('synthetic hand failure')
            return palm()

        def close(self):
            self.closed = True

    ns['LocalHandTracker'] = Tracker
    try:
        session = ns['LocalGestureSession']()
        frame = np.full((24, 32, 3), [10, 40, 90], np.uint8)
        initial = frame.copy()
        sliders = ns['default_light']()
        session.enable()
        for _ in range(3):
            light, info = session.update(frame, sliders)
        assert info['mode'] == 'gesture' and info['status'].startswith('Tracking palm')
        assert info['landmarks'] is not None and info['hand_ms'] >= 0
        np.testing.assert_array_equal(frame, initial)
        np.testing.assert_array_equal(created[-1].frames[-1], initial)
        snapshot = dict(light)
        sliders.update(x=3, power=0.0, specular=1.0)
        next_light, _ = session.update(frame, sliders)
        assert next_light['x'] != 3 and next_light['power'] == snapshot['power'] and next_light['specular'] == 1
        assert light == snapshot  # Packet state must be a copy.
        session.calibrate()
        assert session.gesture.reference is None
        session.toggle()
        next_light, info = session.update(frame, sliders)
        assert next_light == sliders and info['mode'] == 'manual'
        session.toggle()
        _, info = session.update(np.zeros((30, 40, 3), np.uint8), sliders)
        assert created[0].closed and len(created) == 2 and 'Acquiring' in info['status']
        created[-1].fail = True
        held = dict(session.light)
        light, info = session.update(frame, sliders)  # Shape change recreates tracker.
        created[-1].fail = True
        light, info = session.update(frame, sliders)
        assert info['mode'] == 'manual' and 'synthetic hand failure' in info['status']
        assert created[-1].closed and info['landmarks'] is None
        session.enable()
        assert session.enabled
        session.close()
        session.close()
        assert created[-1].closed

        def missing(path):
            raise RuntimeError('synthetic missing model')
        ns['LocalHandTracker'] = missing
        session = ns['LocalGestureSession']()
        session.enable()
        light, info = session.update(frame, sliders)
        assert light == sliders and 'missing model' in info['status'] and info['mode'] == 'manual'
    finally:
        ns['LocalHandTracker'] = real_tracker


def check_local_controls(ns):
    cv = ns['cv2']
    proxy = SimpleNamespace(**{key: getattr(cv, key) for key in dir(cv)})
    ns['cv2'] = proxy
    sliders, packets = {}, []
    frame = np.full((24, 32, 3), 50, np.uint8)
    window = {'open': True, 'key': -1}

    class Camera:
        thread = SimpleNamespace(is_alive=lambda: False)
        closed = False

        def __init__(self, *args): pass
        def get(self):
            return frame, dict(frame_id=1, age_ms=0)
        def close(self): self.closed = True

    class Tracker:
        closed = False
        def __init__(self, *args): pass
        def detect(self, rgb, now):
            assert rgb is frame
            return palm(.7, .7)
        def close(self): self.closed = True

    ns.update(LIVE_BACKEND='local', HAND_CONTROL=True, CAMERA_INDEX=0,
              CAPTURE_WIDTH=320, CAMERA_FPS=30, LatestLocalCamera=Camera,
              LocalHandTracker=Tracker)
    proxy.namedWindow = lambda *a: None
    proxy.resizeWindow = lambda *a: None
    proxy.createTrackbar = lambda label, name, value, *a: sliders.update({label: value})
    proxy.setTrackbarPos = lambda label, name, value: sliders.update({label: value})
    proxy.getTrackbarPos = lambda label, name: sliders[label]
    proxy.getWindowProperty = lambda *a: float(window['open'])
    proxy.destroyWindow = lambda *a: window.update(open=False)
    proxy.waitKey = lambda *a: window['key']
    proxy.imshow = lambda *a: None
    try:
        assert ns['start_webcam']()
        camera, session = ns['_local_camera'], ns['_local_gesture']
        for _ in range(3):
            packets.append(ns['get_frame']())
        rgb, light, info = packets[-1]
        assert rgb is frame and info['frame_id'] == 1 and info['mode'] == 'gesture'
        assert light['x'] > -.6 and light['y'] > -.4
        assert abs((-3 + sliders['X'] * .05) - light['x']) <= .025
        window['key'] = ord('g')
        ns['show_local_frame'](frame)
        sliders['X'] = 100
        assert ns['get_frame']()[1]['x'] == 2
        ns['show_local_frame'](frame)  # G enables again.
        window['key'] = ord('c')
        ns['show_local_frame'](frame)
        assert session.gesture.reference is None
        window['key'] = ord('r')
        ns['show_local_frame'](frame)
        assert session.light == ns['default_light']()
        tracker = session.tracker
        ns['stop_webcam']()
        assert camera.closed and tracker.closed and not window['open']
        assert ns['_local_camera'] is None and ns['_local_gesture'] is None
    finally:
        ns['cv2'] = cv


def finger_hand(ratio):
    points = palm(y=.65)
    points[0].y = .8
    for pip, tip, x in ((6, 8, .43), (10, 12, .48), (14, 16, .53), (18, 20, .58)):
        points[pip].x, points[pip].y = x, .6
        points[tip].x = .5 + (x - .5) * ratio
        points[tip].y = .8 - .2 * ratio
    return points


def check_intensity(ns):
    openness = ns['hand_openness']
    assert openness(finger_hand(.7), 640, 480) == 0
    assert openness(finger_hand(1.7), 640, 480) == 1
    np.testing.assert_allclose(openness(finger_hand(1.25), 640, 480), .5)
    samples = [openness(finger_hand(r), 640, 480) for r in np.linspace(.7, 1.7, 21)]
    assert all(b >= a for a, b in zip(samples, samples[1:]))
    # Translation, apparent distance and aspect ratio do not change the ratio.
    for scale in (.5, 1.1):
        points = finger_hand(1.25)
        for p in points:
            p.x = .45 + (p.x - .5) * scale
            p.y = .4 + (p.y - .5) * scale
        for width, height in ((640, 480), (320, 240), (640, 360)):
            np.testing.assert_allclose(openness(points, width, height), .5)
    for bad in (None, [], palm()):
        assert openness(bad, 640, 480) is None
    for bad_value in (-.1, float('nan')):
        points = finger_hand(1.7)
        points[8].x = bad_value
        assert openness(points, 640, 480) is None

    control, light = ns['PalmLightControl'](), ns['default_light']()
    for i in range(100):
        control.update(finger_hand(1.7), 640, 480, i * .05, light)
    assert 11.99 < light['power'] <= 12
    before = light['power']
    control.update(finger_hand(.7), 640, 480, 5.0, light)
    assert 0 < light['power'] < before  # Smooth dimming, not an instant switch.
    for i in range(1, 100):
        control.update(finger_hand(.7), 640, 480, 5 + i * .05, light)
    assert 0 <= light['power'] < .001
    for i in range(100):
        control.update(finger_hand(1.25), 640, 480, 10 + i * .05, light)
    np.testing.assert_allclose(light['power'], 6, atol=.001)
    held = light['power']
    points = finger_hand(1.7)
    points[8].x = -.1
    assert 'intensity held' in control.update(points, 640, 480, 15, light)
    assert light['power'] == held
    control.update(None, 640, 480, 15.05, light)
    assert light['power'] == held
    for t in (15.1, 15.15):
        control.update(finger_hand(.7), 640, 480, t, light)
        assert light['power'] == held


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-smoke', action='store_true')
    parser.add_argument('--download-model', action='store_true')
    args = parser.parse_args()
    nb, ns = load_helpers()
    check_mapping(ns)
    check_intensity(ns)
    check_session(ns)
    # Fresh namespace so fake lifecycle objects do not affect real model smoke.
    check_local_controls(load_helpers()[1])
    if args.download_model:
        source = next(''.join(c['source']) for c in nb['cells'] if 'hand-model' in c['metadata'].get('tags', []))
        tree = ast.parse(source)
        tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
        exec(compile(tree, 'notebook hand model setup', 'exec'), ns)
        ns['prepare_hand_model']('models/hand_landmarker.task')
    if args.model_smoke:
        tracker = ns['LocalHandTracker'](Path('models/hand_landmarker.task'))
        try:
            for t in (1., 1., 1.01):
                assert tracker.detect(np.zeros((240, 320, 3), np.uint8), t) is None
            assert tracker.last_timestamp == 1010
        finally:
            tracker.close()
        print('Real MediaPipe VIDEO smoke passed on blank RGB frames (including repeated input times).')
    print('Local Level 3 checks passed: mapping, smoothing, calibration, loss/reacquisition, bounds, '
          'aspect ratio, frame/control alignment, G/C/R, manual fallback, errors and cleanup.')


if __name__ == '__main__':
    main()
