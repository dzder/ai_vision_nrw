def hand_openness(landmarks, width, height):
    """Projected finger extension, normalized by wrist-to-PIP length.

    Average four fingers; thumb position does not change intensity. Both axes
    use image-width units so distance and camera aspect ratio cancel out.
    Invalid/clipped landmarks hold intensity rather than switching it off.
    """
    if landmarks is None or len(landmarks) < 21:
        return None
    indices = (0, 6, 8, 10, 12, 14, 16, 18, 20)
    xy = np.asarray([[landmarks[i].x, landmarks[i].y] for i in indices], dtype=np.float64)
    if not np.isfinite(xy).all() or (xy < 0).any() or (xy > 1).any():
        return None
    xy[:, 1] *= height / width
    pip_distance = np.linalg.norm(xy[1::2] - xy[0], axis=1)
    if (pip_distance < 1e-6).any():
        return None
    tip_distance = np.linalg.norm(xy[2::2] - xy[0], axis=1)
    extension = np.clip((tip_distance / pip_distance - 0.9) / 0.7, 0, 1).mean()
    return float(extension * extension * (3 - 2 * extension))


class PalmLightControl:
    """Map a front-facing palm to relative XYZ; never interpret landmark Z as meters."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.reference = None
        self.last_time = None
        self.streak = 0
        self.shape = None

    def update(self, landmarks, width, height, now, light):
        if self.shape != (height, width):
            self.reset()
            self.shape = (height, width)
        palm = None
        if landmarks is not None and len(landmarks) >= 21:
            palm = np.asarray([[landmarks[i].x, landmarks[i].y]
                               for i in (0, 5, 9, 13, 17)], dtype=np.float64)
        if (palm is None or not np.isfinite(palm).all()
                or (palm < 0).any() or (palm > 1).any()):
            self.last_time, self.streak = None, 0
            return "No valid palm - holding light"
        span = float(np.hypot(palm[1, 0] - palm[4, 0],
                              (palm[1, 1] - palm[4, 1]) * height / width))
        if span < 0.025:
            self.last_time, self.streak = None, 0
            return "Show a larger front-facing palm - holding light"
        if self.last_time is not None and now - self.last_time > 0.5:
            self.streak = 0
        dt = 0.05 if self.last_time is None else np.clip(now - self.last_time, 0, 0.1)
        self.last_time = now
        self.streak += 1
        if self.streak < 3:
            return "Acquiring palm - holding light"
        if self.reference is None:
            self.reference = (span, light['z'])
        u, v = palm.mean(axis=0)
        target = (np.clip(6 * (u - 0.5), -3, 3),
                  np.clip(4 * (v - 0.5), -2, 2),
                  np.clip(self.reference[1] - 1.5 * np.log(span / self.reference[0]), -2, 0.7))
        alpha = 1 - np.exp(-dt / 0.12)
        for key, value in zip(('x', 'y', 'z'), target):
            light[key] = float(light[key] + alpha * (value - light[key]))
        openness = hand_openness(landmarks, width, height)
        if openness is None:
            return "Tracking palm | intensity held"
        # Closed -> 0, open -> 12; ambient scene lighting remains visible.
        power_alpha = 1 - np.exp(-dt / 0.18)
        light['power'] = float(np.clip(light['power'] + power_alpha *
                                      (12.0 * openness - light['power']), 0, 12))
        return f"Tracking palm | open {openness:.0%} | power {light['power']:.1f}"


class LocalHandTracker:
    """Synchronous VIDEO detection on the exact RGB frame used for depth."""
    def __init__(self, model_path):
        from pathlib import Path
        try:
            import mediapipe as mp
        except ImportError as error:
            raise RuntimeError('Install requirements-notebook.txt and restart the kernel') from error
        if not Path(model_path).is_file():
            raise RuntimeError('Hand model missing; run the hand-model setup cell')
        self.mp = mp
        self.last_timestamp = -1
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path),
                                             delegate=mp.tasks.BaseOptions.Delegate.CPU),
            running_mode=mp.tasks.vision.RunningMode.VIDEO, num_hands=1,
            min_hand_detection_confidence=0.6, min_hand_presence_confidence=0.6,
            min_tracking_confidence=0.6,
        )
        self.tracker = mp.tasks.vision.HandLandmarker.create_from_options(options)

    def detect(self, rgb, now):
        # Millisecond rounding must not create duplicate timestamps.
        timestamp = max(self.last_timestamp + 1, int(now * 1000))
        self.last_timestamp = timestamp
        image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB,
                              data=np.ascontiguousarray(rgb))
        result = self.tracker.detect_for_video(image, timestamp)
        return result.hand_landmarks[0] if result.hand_landmarks else None

    def close(self):
        if self.tracker is not None:
            self.tracker.close()
            self.tracker = None


class LocalGestureSession:
    """Own light/calibration state and fail back to manual controls on errors."""
    def __init__(self):
        self.light = default_light()
        self.gesture = PalmLightControl()
        self.tracker = None
        self.enabled = False
        self.status = 'Manual XYZ sliders'
        self.shape = None

    def enable(self):
        try:
            if self.tracker is None:
                self.tracker = LocalHandTracker(globals().get('HAND_MODEL_PATH', 'models/hand_landmarker.task'))
            self.enabled = True
            self.calibrate()
        except Exception as error:
            self.enabled = False
            self.status = f'Hand tracking unavailable: {error}. G: retry; sliders active'
            print(self.status)

    def calibrate(self):
        self.gesture.reset()
        self.status = 'Show one palm to calibrate Z' if self.enabled else 'Manual XYZ sliders'

    def toggle(self):
        if self.enabled:
            self.enabled = False
            self.calibrate()
        else:
            self.enable()

    def update(self, rgb, sliders):
        start = time.perf_counter()
        landmarks = None
        if self.enabled:
            self.light.update(specular=sliders['specular'])
            try:
                # Recreate VIDEO tracking after a resolution change, not just the Z reference.
                if self.shape is not None and self.shape != rgb.shape[:2]:
                    self.close()
                    self.enable()
                self.shape = rgb.shape[:2]
                if self.enabled:
                    landmarks = self.tracker.detect(rgb, start)
                    self.status = self.gesture.update(landmarks, rgb.shape[1], rgb.shape[0],
                                                      start, self.light)
            except Exception as error:
                self.enabled = False
                try:
                    self.close()
                except Exception as close_error:
                    print(f'Hand tracker cleanup failed: {close_error}')
                self.status = f'Tracking failed: {error}. G: retry; sliders active'
                print(self.status)
                landmarks = None
        else:
            self.light = dict(sliders)
        return dict(self.light), dict(mode='gesture' if self.enabled else 'manual',
                                      status=self.status, landmarks=landmarks,
                                      hand_ms=(time.perf_counter() - start) * 1000)

    def close(self):
        self.enabled = False
        if self.tracker is not None:
            self.tracker.close()
            self.tracker = None


def sync_local_controls(light):
    for key, label, low, high, step in local_light_specs():
        if key in ('x', 'y', 'z', 'power'):
            cv2.setTrackbarPos(label, _local_window, round((light[key] - low) / step))


def annotate_local_hand(picture, hand, height, width):
    """Draw only on the output copy, keeping overlays out of model inputs."""
    points = hand.get('landmarks')
    if globals().get('SHOW_HAND_LANDMARKS', True) and points is not None:
        ox, oy = (width, height + 64) if SHOW_DIAGNOSTICS else (0, 0)
        for p in points:
            if np.isfinite([p.x, p.y]).all() and 0 <= p.x <= 1 and 0 <= p.y <= 1:
                cv2.circle(picture, (ox + round(p.x * (width - 1)), oy + round(p.y * (height - 1))),
                           2, (0, 255, 160), -1)
    cv2.putText(picture, hand['status'][:90], (8, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 160), 1, cv2.LINE_AA)



