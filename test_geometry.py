"""Synthetic checks requiring NumPy/OpenCV, but no model download or camera."""
import contextlib
import io
import sys
import types
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from normals import compute_normals, sample_normal
from probe import PointProbe


class GeometryTests(unittest.TestCase):
    def test_flat_surface_and_distance_offset(self):
        depth = np.full((40, 60), 0.2, dtype=np.float32)
        normals = compute_normals(depth)
        np.testing.assert_allclose(normals[..., :2], 0)
        np.testing.assert_allclose(normals[..., 2], 1)
        np.testing.assert_allclose(compute_normals(depth+0.5), normals)

    def test_sloped_surface_unit_normal_and_patch(self):
        y, x = np.mgrid[:40, :60].astype(np.float32)
        depth = x*0.01+y*0.02
        normals = compute_normals(depth, strength=1)
        expected = np.array([-0.01, -0.02, 1])
        expected /= np.linalg.norm(expected)
        normal, value = sample_normal(depth, normals, (30, 20))
        np.testing.assert_allclose(normal, expected, atol=1e-6)
        self.assertAlmostEqual(value, 0.7, places=6)
        np.testing.assert_allclose(np.linalg.norm(normals, axis=-1), 1, atol=1e-6)
        np.testing.assert_allclose(compute_normals(depth+2, 1), normals, atol=1e-6)
        edge_normal, _ = sample_normal(depth, normals, (59.9, 39.9))
        self.assertAlmostEqual(float(np.linalg.norm(edge_normal)), 1, places=6)

    def test_tracking_translation_and_loss(self):
        random = np.random.default_rng(4)
        frame = random.integers(0, 256, (120, 160, 3), dtype=np.uint8)
        probe = PointProbe()
        probe.update(frame)
        probe.on_mouse(cv2.EVENT_LBUTTONDOWN, 80, 60, 0, None)
        moved = cv2.warpAffine(frame, np.float32([[1, 0, 4], [0, 1, 3]]), (160, 120))
        probe.update(moved)
        self.assertIsNotNone(probe.point)
        np.testing.assert_allclose(probe.point, [84, 63], atol=0.3)
        probe.update(np.zeros_like(frame))
        self.assertIsNone(probe.point)
        self.assertTrue(probe.lost)

    def test_click_mapping_footer_and_clear(self):
        probe = PointProbe()
        probe.update(np.zeros((60, 80, 3), dtype=np.uint8))
        probe.on_mouse(cv2.EVENT_LBUTTONDOWN, 100, 30, 0, None)
        np.testing.assert_equal(probe.point, [20, 30])
        probe.on_mouse(cv2.EVENT_LBUTTONDOWN, 100, 70, 0, None)
        np.testing.assert_equal(probe.point, [20, 30])
        probe.on_mouse(cv2.EVENT_RBUTTONDOWN, 0, 0, 0, None)
        self.assertIsNone(probe.point)

    def test_synchronized_loop_and_benchmark_exit(self):
        import main

        class FakeCamera:
            def __init__(self, *args):
                self.index = 0
                self.closed = False

            def read(self):
                self.index += 1
                return np.full((40, 60, 3), self.index, dtype=np.uint8)

            def close(self):
                self.closed = True

        class FakeEstimator:
            device, dtype, model_shape = "cpu", "float32", (140, 140)

            def __init__(self, *args, **kwargs):
                pass

            def infer(self, frame):
                return np.full(frame.shape[:2], frame[0, 0, 0]/255, dtype=np.float32)

        camera = FakeCamera()
        shown = []
        output = io.StringIO()
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.dict(sys.modules, {"depth": types.SimpleNamespace(DepthEstimator=FakeEstimator)}))
            stack.enter_context(patch.object(main, "Webcam", return_value=camera))
            stack.enter_context(patch.object(sys, "argv", ["main.py", "--level", "1", "--benchmark-frames", "3"]))
            for name in ("namedWindow", "setMouseCallback", "destroyAllWindows"):
                stack.enter_context(patch.object(cv2, name))
            stack.enter_context(patch.object(cv2, "imshow", side_effect=lambda _, value: shown.append(value.copy())))
            stack.enter_context(patch.object(cv2, "waitKey", return_value=-1))
            stack.enter_context(patch.object(cv2, "getWindowProperty", return_value=1))
            stack.enter_context(contextlib.redirect_stdout(output))
            main.main()
        self.assertTrue(camera.closed)
        self.assertEqual(len(shown), 3)
        for index, display in enumerate(shown, 2):
            np.testing.assert_equal(display[:40, :60], index)
            np.testing.assert_equal(display[20, 90], [255, 128, 128])
        self.assertIn("Measured 3 frames:", output.getvalue())


if __name__ == "__main__":
    unittest.main()
