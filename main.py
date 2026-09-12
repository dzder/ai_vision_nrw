"""Level 2 Dynamic Relighting. Run python main.py; Q / Escape exits."""
import argparse
from collections import deque
from time import perf_counter

import cv2
import numpy as np

from capture import Webcam
from normals import compute_normals, normals_to_rgb, sample_normal
from probe import PointProbe
from lighting import LightController, shade, surface_points


def put_label(image, text, position):
    cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (230, 230, 230), 1, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level", type=int, choices=[1, 2], default=2,
                        help="1: geometry/probe only; 2: interactive point-light relighting")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--strength", type=float, default=40.0)
    parser.add_argument("--input-size", type=int, default=518,
                        help="Model resize target, multiple of 14: try 294 for speed, 518 for detail")
    parser.add_argument("--benchmark-frames", type=int, default=0,
                        help="Exit after this many processed frames and print timings (0: unlimited)")
    args = parser.parse_args()
    if min(args.width, args.height) < 3 or not np.isfinite(args.strength) or args.strength < 0:
        parser.error("Dimensions must be >= 3; strength must be finite and nonnegative.")
    if args.input_size < 140 or args.input_size % 14:
        parser.error("--input-size must be a multiple of 14 and at least 140.")
    if args.benchmark_frames < 0:
        parser.error("--benchmark-frames must be nonnegative.")

    camera = None
    title = "Level 2 - Dynamic Relighting" if args.level == 2 else "Level 1 - Geometry Engine"
    totals = np.zeros(5, dtype=np.float64)
    completed = 0
    try:
        from depth import DepthEstimator
        print("Loading Depth Anything V2 Small (first run downloads weights ~99 MB)...", flush=True)
        estimator = DepthEstimator(args.device, input_size=args.input_size)
        print(f"Device: {estimator.device}. Warming up...", flush=True)
        camera = Webcam(args.camera, args.width, args.height)
        warmup = camera.read()
        for _ in range(3):
            estimator.infer(warmup)
        print(f"Actual camera: {warmup.shape[1]}x{warmup.shape[0]}; "
              f"model HxW: {estimator.model_shape}; dtype: {estimator.dtype}", flush=True)
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        probe = PointProbe()
        light = LightController()
        def on_mouse(event, x, y, flags, userdata):
            probe.on_mouse(event, x, y, flags, userdata)
            if args.level == 2:
                light.on_mouse(event, x, y, flags, userdata)
        cv2.setMouseCallback(title, on_mouse)
        durations = deque(maxlen=30)
        fps = 0.0
        while True:
            start = perf_counter()
            frame = camera.read()
            captured = perf_counter()
            depth = estimator.infer(frame)
            inferred = perf_counter()
            normals = compute_normals(depth, args.strength)
            normal_bgr = cv2.cvtColor(normals_to_rgb(normals), cv2.COLOR_RGB2BGR)
            probe.update(frame)
            geometry_done = perf_counter()
            h, w = frame.shape[:2]
            light.shape = (h, w)
            if args.level == 2:
                points = surface_points(depth, args.strength)
                relit = shade(frame, points, normals, light.position,
                              intensity=light.intensity, specular=light.specular)
            shaded = perf_counter()
            panels = 3 if args.level == 2 else 2
            display = np.zeros((h+200, max(panels*w, 1100), 3), dtype=np.uint8)
            display[:h, :w] = frame
            display[:h, w:2*w] = normal_bgr
            if args.level == 2:
                display[:h, 2*w:3*w] = relit
                light.draw(display)
            panel_label = "Camera | Normals XYZ->RGB" + (" | Relit" if args.level == 2 else "")
            lines = [f"{panel_label}    FPS: {fps:.1f}    {estimator.device}    "
                     f"model HxW: {estimator.model_shape}",
                     "Click camera/normals to probe; right-click/C: clear; B: reset reference; Q: quit"]
            if args.level == 2:
                lines += ["Move mouse over Relit: light XY | wheel/[/]: Z toward/away from scene | +/-: power | S: specular | R: reset",
                          f"Light XYZ: ({light.position[0]:+.2f}, {light.position[1]:+.2f}, {light.position[2]:+.2f}) proxy units | "
                          f"power: {light.intensity:.1f} | specular: {'on' if light.specular else 'off'}"]
            if probe.point is not None:
                normal, relative_depth = sample_normal(depth, normals, probe.point)
                if probe.baseline is None:
                    probe.baseline = normal.copy()
                angle = np.degrees(np.arccos(np.clip(np.dot(normal, probe.baseline), -1, 1)))
                probe.draw(display, w, normal)
                lines += [f"N = ({normal[0]:+.3f}, {normal[1]:+.3f}, {normal[2]:+.3f})    "
                          f"|N| = {np.linalg.norm(normal):.3f}    direction change: {angle:.1f} deg",
                          f"Relative depth: {relative_depth:.3f} (0 near, 1 far; per-frame scale, NOT metres)"]
            else:
                lines += ["Tracking lost: click the object again." if probe.lost else
                          "No point selected. Move the object slowly after selecting a textured feature.",
                          "Normals have unit length at every distance; relative depth cannot measure travel distance."]
            if durations:
                avg = np.mean(durations, axis=0)*1000
                lines += [f"Previous frames: capture {avg[0]:.1f} ms | depth {avg[1]:.1f} ms | "
                          f"normals/probe {avg[2]:.1f} ms | lighting {avg[3]:.1f} ms | display {avg[4]:.1f} ms"]
            for index, line in enumerate(lines):
                put_label(display, line, (10, h+23+index*26))
            cv2.imshow(title, display)
            key = cv2.waitKey(1) & 0xFF
            ended = perf_counter()
            timing = np.array([captured-start, inferred-captured,
                               geometry_done-inferred, shaded-geometry_done, ended-shaded])
            durations.append(timing)
            totals += timing
            completed += 1
            # Previous completed frames: includes capture, CPU/GPU transfer,
            # inference, normals, display submission and GUI event processing.
            fps = len(durations) / np.sum(durations)
            if args.level == 2:
                light.on_key(key)
            if key in (ord("c"), ord("C")):
                probe.clear()
            elif key in (ord("b"), ord("B")):
                probe.baseline = None
            if key in (ord("q"), ord("Q"), 27) or cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                break
            if args.benchmark_frames and completed >= args.benchmark_frames:
                break
    except KeyboardInterrupt:
        pass
    finally:
        if camera is not None:
            camera.close()
        cv2.destroyAllWindows()
        if completed:
            averages = totals/completed*1000
            print(f"Measured {completed} frames: {completed/totals.sum():.2f} FPS; "
                  f"capture/depth/normals+probe/lighting/display ms: {averages.round(2).tolist()}")


if __name__ == "__main__":
    main()
