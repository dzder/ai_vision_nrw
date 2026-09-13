"""Renderer and application checks without camera access or model downloads."""
import contextlib
import io
import sys
import types
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from lighting import LightController, shade, surface_points
from normals import compute_normals


class LightingTests(unittest.TestCase):
    def setUp(self):
        self.depth = np.full((41, 61), 0.5, dtype=np.float32)
        self.frame = np.full((41, 61, 3), 100, dtype=np.uint8)
        self.points = surface_points(self.depth)
        self.normals = compute_normals(self.depth)

    def render(self, light, **kwargs):
        return shade(self.frame, self.points, self.normals, light, **kwargs)

    def test_front_back_and_falloff(self):
        near = self.render([0, 0, -0.25], ambient=0, specular=0)
        far = self.render([0, 0, -2], ambient=0, specular=0)
        behind = self.render([0, 0, 2], ambient=0)
        self.assertGreater(int(near[20, 30, 0]), int(far[20, 30, 0]))
        np.testing.assert_equal(behind, 0)
        self.assertEqual(near.dtype, np.uint8)

    def test_moving_light_moves_brightness_and_highlight(self):
        left = self.render([-0.3, 0, -0.3])
        right = self.render([0.3, 0, -0.3])
        np.testing.assert_allclose(left, right[:, ::-1], atol=1)
        self.assertGreater(int(left[20, 12, 0]), int(left[20, 48, 0]))
        diffuse = self.render([0, 0, -0.3], specular=0)
        glossy = self.render([0, 0, -0.3], specular=0.35)
        delta = glossy.astype(float)-diffuse
        self.assertGreater(delta[20, 30, 0], delta[20, 0, 0])

    def test_linear_color_round_trip_and_zero_power(self):
        self.frame[:] = [37, 128, 210]
        np.testing.assert_equal(self.render([0, 0, -1], ambient=1, intensity=0), self.frame)
        np.testing.assert_equal(self.render([0, 0, -1], ambient=0, intensity=0), 0)

    def test_surface_tangents_match_normals(self):
        y, x = np.mgrid[:41, :61].astype(np.float32)
        depth = x*0.01+y*0.004
        points = surface_points(depth, 40)
        tangent_x = points[20, 31]-points[20, 29]
        tangent_y = points[21, 30]-points[19, 30]
        normal = np.cross(tangent_x, tangent_y)
        normal /= np.linalg.norm(normal)
        np.testing.assert_allclose(normal, compute_normals(depth, 40)[20, 30], atol=1e-6)

    def test_degenerate_and_invalid_inputs(self):
        coincident = self.render(self.points[20, 30], ambient=0)
        np.testing.assert_equal(coincident[20, 30], 0)
        for kwargs in ({"intensity": -1}, {"ambient": np.nan}, {"shininess": 0}):
            with self.assertRaises(ValueError):
                self.render([0, 0, -1], **kwargs)
        with self.assertRaises(ValueError):
            self.render([np.nan, 0, -1])
        with self.assertRaises(ValueError):
            shade(self.frame, self.points[:-1], self.normals, [0, 0, -1])

    def test_mouse_keyboard_and_bounds(self):
        control = LightController()
        control.shape = (41, 61)
        original = control.position.copy()
        control.on_mouse(cv2.EVENT_MOUSEMOVE, 30, 20, 0, None)
        np.testing.assert_equal(control.position, original)
        control.on_mouse(cv2.EVENT_MOUSEMOVE, 152, 20, 0, None)
        np.testing.assert_equal(control.position[:2], [0, 0])
        control.on_mouse(cv2.EVENT_MOUSEWHEEL, 152, 20, 120 << 16, None)
        self.assertAlmostEqual(float(control.position[2]), -0.65, places=6)
        control.on_mouse(cv2.EVENT_MOUSEWHEEL, 152, 20, (-120 & 0xffff) << 16, None)
        self.assertAlmostEqual(float(control.position[2]), -0.75, places=6)
        control.on_key(ord("s"))
        self.assertEqual(control.specular, 0)
        for _ in range(100):
            control.on_key(ord("]"))
            control.on_key(ord("-"))
        self.assertAlmostEqual(float(control.position[2]), -0.05, places=6)
        self.assertEqual(control.intensity, 0)
        control.on_key(ord("r"))
        np.testing.assert_equal(control.position, original)

    def test_level_two_loop_and_error_cleanup(self):
        import main

        class Camera:
            def __init__(self):
                self.count = 0
                self.closed = False

            def read(self):
                self.count += 1
                # Exercise a size change after one completed frame.
                shape = (41, 61, 3) if self.count < 3 else (45, 65, 3)
                return np.full(shape, 60+self.count, dtype=np.uint8)

            def close(self):
                self.closed = True

        class Estimator:
            device, dtype, model_shape = "cpu", "float32", (140, 140)

            def __init__(self, *args, **kwargs):
                pass

            def infer(self, frame):
                return np.full(frame.shape[:2], frame[0, 0, 0]/255, dtype=np.float32)

        for fail in (False, True):
            camera = Camera()
            shown, renders, callbacks = [], [], []

            def render(frame, points, normals, position, **kwargs):
                if fail:
                    raise RuntimeError("render failure")
                result = shade(frame, points, normals, position, **kwargs)
                renders.append(result.copy())
                return result

            def show(title, display):
                shown.append(display.copy())
                # Move the light, then click the camera to test callback routing.
                h, w = renders[-1].shape[:2]
                callbacks[0](cv2.EVENT_MOUSEMOVE, 2*w+20, 20, 0, None)
                callbacks[0](cv2.EVENT_LBUTTONDOWN, 20, 20, 0, None)

            with contextlib.ExitStack() as stack:
                stack.enter_context(patch.dict(sys.modules, {"depth": types.SimpleNamespace(DepthEstimator=Estimator)}))
                stack.enter_context(patch.object(main, "Webcam", return_value=camera))
                stack.enter_context(patch.object(main, "shade", side_effect=render))
                stack.enter_context(patch.object(sys, "argv", ["main.py", "--benchmark-frames", "2"]))
                stack.enter_context(patch.object(cv2, "namedWindow"))
                cleanup = stack.enter_context(patch.object(cv2, "destroyAllWindows"))
                stack.enter_context(patch.object(cv2, "setMouseCallback", side_effect=lambda _, cb: callbacks.append(cb)))
                stack.enter_context(patch.object(cv2, "imshow", side_effect=show))
                stack.enter_context(patch.object(cv2, "waitKey", return_value=ord("]")))
                stack.enter_context(patch.object(cv2, "getWindowProperty", return_value=1))
                stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                if fail:
                    with self.assertRaisesRegex(RuntimeError, "render failure"):
                        main.main()
                else:
                    main.main()
                cleanup.assert_called_once()
            self.assertTrue(camera.closed)
            self.assertEqual(len(shown), 0 if fail else 2)
            for index, (display, relit) in enumerate(zip(shown, renders), 2):
                h, w = relit.shape[:2]
                np.testing.assert_equal(display[0, :w], 60+index)
                # Bottom row avoids the interactive light marker and the cull
                # ring (both centered on the light's screen column); compare the
                # unaffected right-hand segment.
                np.testing.assert_equal(display[h-1, 2*w+w//2:3*w], relit[h-1, w//2:])


if __name__ == "__main__":
    unittest.main()
