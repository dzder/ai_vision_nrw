"""Exercise local capture/window lifecycle with a fake camera, without opening GUI."""
import queue
from types import SimpleNamespace

import cv2
import numpy as np

from test_notebook_performance import load_helpers


class FakeCapture:
    def __init__(self, opened=True):
        self.opened = opened
        self.frames = queue.Queue()
        self.released = False

    def isOpened(self):
        return self.opened

    def set(self, *args):
        return True

    def read(self):
        frame = self.frames.get()
        return frame is not None, frame

    def release(self):
        self.released = True
        self.frames.put(None)


def main():
    _, ns = load_helpers()
    proxy = SimpleNamespace(**{key: getattr(cv2, key) for key in dir(cv2)})
    ns['cv2'] = proxy
    capture = FakeCapture()
    proxy.VideoCapture = lambda *a: capture
    camera = ns['LatestLocalCamera'](width=10)
    try:
        for i in range(30):
            capture.frames.put(np.full((12, 20, 3), [0, 0, i], np.uint8))
        with camera.condition:
            assert camera.condition.wait_for(lambda: camera.sequence == 30, timeout=2)
        rgb, info = camera.get()
        assert info['frame_id'] == 30 and info['age_ms'] >= 0
        assert rgb.shape == (6, 10, 3)
        np.testing.assert_array_equal(rgb[..., 0], 29)
        np.testing.assert_array_equal(rgb[..., 2], 0)  # Correct BGR -> RGB.
        try:
            camera.get(timeout=0.01)
        except RuntimeError as error:
            assert 'Timed out' in str(error)
        else:
            raise AssertionError('The same frame was delivered twice')
        capture.frames.put(np.full((9, 15, 3), [0, 0, 45], np.uint8))
        rgb, info = camera.get()
        assert info['frame_id'] == 31 and rgb.shape == (6, 10, 3)
    finally:
        camera.close()
    assert capture.released and not camera.thread.is_alive()
    assert camera.get() is None

    capture = FakeCapture()
    camera = ns['LatestLocalCamera']()
    capture.frames.put(None)
    try:
        camera.get()
    except RuntimeError as error:
        assert 'stopped returning frames' in str(error)
    else:
        raise AssertionError('Camera read failure was ignored')
    finally:
        camera.close()
    assert capture.released

    capture = FakeCapture(opened=False)
    try:
        ns['LatestLocalCamera']()
    except RuntimeError as error:
        assert 'Could not open camera' in str(error)
    else:
        raise AssertionError('Unavailable camera accepted')
    assert capture.released

    # Real notebook lifecycle functions, fake UI and camera only.
    sliders, shown = {}, []
    window = dict(visible=True)
    proxy.namedWindow = lambda *a: window.update(visible=True)
    proxy.resizeWindow = lambda *a: None
    proxy.createTrackbar = lambda label, name, value, maximum, callback: sliders.update({label: value})
    proxy.setTrackbarPos = lambda label, name, value: sliders.update({label: value})
    proxy.getTrackbarPos = lambda label, name: sliders[label]
    proxy.getWindowProperty = lambda *a: float(window['visible'])
    proxy.destroyWindow = lambda *a: window.update(visible=False)
    proxy.waitKey = lambda *a: -1
    proxy.imshow = lambda name, bgr: shown.append(bgr.copy())
    ns.update(LIVE_BACKEND='local', CAMERA_INDEX=0, CAPTURE_WIDTH=320, CAMERA_FPS=30)
    capture = FakeCapture()
    assert ns['start_webcam']()
    capture.frames.put(np.full((12, 20, 3), [10, 30, 80], np.uint8))
    rgb, light, info = ns['get_frame']()
    for key, value in ns['default_light']().items():
        assert abs(light[key]-value) < 1e-6
    assert ns['show_local_frame'](rgb)
    np.testing.assert_array_equal(shown[-1], np.full((12, 20, 3), [10, 30, 80], np.uint8))
    sliders['X'] = 0
    proxy.waitKey = lambda *a: ord('r')
    assert ns['show_local_frame'](rgb)
    assert sliders['X'] == 48
    for key, setting, default in (('s', 'SHADOWS_ENABLED', True), ('v', 'SHOW_SHADOW_VISIBILITY', False)):
        proxy.waitKey = lambda *a, key=key: ord(key)
        assert ns['show_local_frame'](rgb)
        assert ns[setting] is not default
        assert ns['show_local_frame'](rgb)
        assert ns[setting] is default
    proxy.waitKey = lambda *a: ord('q')
    assert not ns['show_local_frame'](rgb)
    window['visible'] = False
    assert ns['get_frame']() is None and not ns['show_local_frame'](rgb)
    ns['stop_webcam']()
    assert capture.released and ns['_local_camera'] is None

    # Full local loop must use raw display and never JPEG-encode/send panels.
    handles, stopped = [], []
    class Handle:
        def __init__(self): self.updates = []
        def update(self, value): self.updates.append(value)
    def display(*a, **kw):
        h = Handle()
        handles.append(h)
        return h
    frames = [np.full((24, 32, 3), i*20, np.uint8) for i in range(5)]
    packets = iter([(f, ns['default_light'](), dict(mode='manual', status='Local', age_ms=0)) for f in frames] + [None])
    def no_jpeg(*a): raise AssertionError('Local loop encoded JPEG')
    def no_transport(*a): raise AssertionError('Local loop invoked Colab')
    ns.update(display=display, Markdown=lambda x: x, DEVICE='cpu', ASSUMED_HFOV_DEG=60,
              PROFILE_STAGES=True, SHOW_DIAGNOSTICS=False, STATUS_INTERVAL=0,
              start_webcam=lambda: True, get_frame=lambda: next(packets),
              stop_webcam=lambda: stopped.append(True),
              encode_live_image=no_jpeg, start_colab_webcam=no_transport,
              infer_relative_inverse=lambda f, **kw: ns['torch'].ones(f.shape[:2]),
              show_local_frame=lambda rgb: shown.append(rgb.copy()) or True)
    before = len(shown)
    ns['run_live']()
    assert len(handles) == 1 and len(shown)-before == 5 and stopped == [True]
    assert 'cycles/s' in handles[0].updates[-1]
    check_shadow_loop()
    print('Local notebook checks passed: latest-frame capture, no duplicates/backlog, RGB/aspect, sliders/reset, raw display, quit/close, and failure cleanup.')


def check_shadow_loop():
    """Exercise shadow integration, exact packet alignment and all display modes."""
    _, ns = load_helpers()
    torch = ns['torch']
    proxy = SimpleNamespace(**{key: getattr(cv2, key) for key in dir(cv2)})
    labels = []
    proxy.putText = lambda picture, text, *a: labels.append(text) or picture
    class Handle:
        def __init__(self): self.updates = []
        def update(self, value): self.updates.append(value)
    stopped, shown, statuses = [], [], []
    def display(*a, **kw):
        handle = Handle()
        statuses.append(handle)
        return handle
    frames = [np.full((*shape, 3), 30+i, np.uint8)
              for i, shape in enumerate(((24, 32), (24, 32), (30, 40), (30, 40)))]
    lights = [dict(ns['default_light'](), x=-.6+i*.1, power=4. if i < 3 else 0.) for i in range(4)]
    frame_lights = [[light, dict(light, x=-light['x'])] for light in lights]
    rendered, traced, geometry = [], [], []
    real_render, real_geometry = ns['relight_rgb'], ns['compute_live_geometry']
    def compute_geometry(raw, mapper, estimator):
        result = real_geometry(raw, mapper, estimator)
        geometry.append((int(raw[0, 0]), result, estimator.intrinsics))
        return result
    def shadow(depth, normals, valid, light, intrinsics, **kw):
        marker, result, expected_intrinsics = geometry[-1]
        assert depth is result[0] and normals is result[1] and valid is result[3]
        assert intrinsics == expected_intrinsics and light in frame_lights[marker-30]
        traced.append(marker)
        return torch.full_like(depth, (marker-29)*.25)
    def render(frame, n, p, valid, light, **kw):
        assert frame[0, 0, 0] == geometry[-1][0]
        assert light == frame_lights[int(frame[0, 0, 0])-30]
        result = real_render(frame, n, p, valid, light, **kw)
        rendered.append((result.cpu().numpy(), kw['visibility'][0].cpu().numpy()))
        return result
    ns.update(cv2=proxy, LIVE_BACKEND='local', DEVICE='cpu', ASSUMED_HFOV_DEG=60,
        PROFILE_STAGES=True, STATUS_INTERVAL=0, SHOW_HAND_LANDMARKS=False,
        start_webcam=lambda: True, stop_webcam=lambda: stopped.append(True),
        display=display, Markdown=lambda x: x,
        infer_relative_inverse=lambda frame, **kw: torch.full(frame.shape[:2], float(frame[0, 0, 0])),
        compute_live_geometry=compute_geometry, compute_shadow_visibility=shadow, relight_rgb=render,
        annotate_local_hand=lambda *a: None,
        show_local_frame=lambda picture: shown.append(picture.copy()) or True)
    for enabled, diagnostics, visibility_view in ((True, False, False), (True, True, False),
                                                  (True, True, True), (False, False, False),
                                                  (False, False, True)):
        packets = iter([(f, light, dict(mode='gesture', status='Tracking', age_ms=0, lights=group))
                        for f, light, group in zip(frames, lights, frame_lights)] + [None])
        ns.update(SHADOWS_ENABLED=enabled, SHOW_DIAGNOSTICS=diagnostics,
                  SHOW_SHADOW_VISIBILITY=visibility_view, get_frame=lambda: next(packets))
        traced.clear()
        ns['run_live']()
        assert traced == ([30, 30, 31, 31, 32, 32] if enabled else [])  # Each powered light traces.
        for f, actual, (relit, visibility) in zip(frames, shown[-4:], rendered[-4:]):
            h, w = f.shape[:2]
            if visibility_view:
                expected = np.repeat(np.rint(visibility[..., None]*255).astype(np.uint8), 3, axis=-1)
                np.testing.assert_array_equal(actual, expected)
            elif diagnostics:
                assert actual.shape == (2*(h+32), 2*w, 3)
                np.testing.assert_array_equal(actual[32:h+32, :w], f)
                np.testing.assert_array_equal(actual[h+64:, w:], relit)
            else:
                np.testing.assert_array_equal(actual, relit)
        assert any('**Shadows:**' in s for s in statuses[-1].updates)
        overlays = [s for s in labels if 'shadows ' in s]
        assert all(f"shadows {'on' if enabled else 'off'}" in s for s in overlays[-4:])
    assert len(stopped) == 5
    def fail_shadow(*a, **kw):
        raise RuntimeError('synthetic shadow failure')
    ns.update(SHADOWS_ENABLED=True, compute_shadow_visibility=fail_shadow,
              get_frame=lambda: (frames[0], lights[0], dict(mode='manual', status='Local', age_ms=0)))
    try:
        ns['run_live']()
    except RuntimeError as error:
        assert str(error) == 'synthetic shadow failure'
    else:
        raise AssertionError('Shadow failure was swallowed')
    assert len(stopped) == 6
    print('Local shadows passed: frame/light/geometry alignment, S/V, zero power, disabled, diagnostics, resize, timing/status and error cleanup.')


if __name__ == '__main__':
    main()
