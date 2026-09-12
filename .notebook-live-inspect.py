def start_colab_webcam():
    from google.colab.output import eval_js
    display(Javascript(r"""
// Browser-local tracking keeps hand inference independent of Colab depth inference.
if (window.l1StopCamera) window.l1StopCamera();
window.l1Camera = {};

window.L3GestureControl = class {
    constructor() { this.reset(); }
    reset() {
        this.reference = null;
        this.lastTime = null;
        this.streak = 0;
    }
    update(landmarks, width, height, now, light) {
        const palm = landmarks && [0, 5, 9, 13, 17].map(i => landmarks[i]);
        if (!palm || palm.some(p => !p || !Number.isFinite(p.x) || !Number.isFinite(p.y)
                || p.x < 0 || p.x > 1 || p.y < 0 || p.y > 1)) {
            this.lastTime = null;
            this.streak = 0;
            return 'No hand — holding light';
        }
        // Normalize both axes by image width, so aspect ratio cannot distort size.
        const span = Math.hypot(palm[1].x - palm[4].x,
                               (palm[1].y - palm[4].y) * height / width);
        if (!Number.isFinite(span) || span < 0.025) {
            this.lastTime = null;
            this.streak = 0;
            return 'Show a larger, front-facing palm — holding light';
        }
        if (this.lastTime !== null && now - this.lastTime > 500) this.streak = 0;
        const dt = this.lastTime === null ? 50 : Math.max(0, Math.min(now - this.lastTime, 100));
        this.lastTime = now;
        if (++this.streak < 3) return 'Acquiring hand — holding light';
        if (!this.reference) this.reference = {span, z: light.z};
        const u = palm.reduce((sum, p) => sum + p.x, 0) / palm.length;
        const v = palm.reduce((sum, p) => sum + p.y, 0) / palm.length;
        const clamp = (x, lo, hi) => Math.max(lo, Math.min(hi, x));
        const target = {
            x: clamp((u - 0.5) * 6, -3, 3),
            y: clamp((v - 0.5) * 4, -2, 2),
            // A larger palm moves the light toward the camera (negative Z).
            z: clamp(this.reference.z - 1.5 * Math.log(span / this.reference.span), -2, 0.7),
        };
        const alpha = 1 - Math.exp(-dt / 120); // Time-based EMA, tau = 120 ms.
        for (const key of ['x', 'y', 'z']) light[key] += alpha * (target[key] - light[key]);
        return 'Tracking hand';
    }
};

window.l3LoadHandTracker = async function() {
    // Pin the JS and WASM to the same release. No Python MediaPipe dependency.
    const root = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.22';
    const {FilesetResolver, HandLandmarker} = await import(root + '/vision_bundle.mjs');
    const files = await FilesetResolver.forVisionTasks(root + '/wasm');
    return HandLandmarker.createFromOptions(files, {
        baseOptions: {
            modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
            delegate: 'CPU',
        },
        runningMode: 'VIDEO', numHands: 1,
        minHandDetectionConfidence: 0.6,
        minHandPresenceConfidence: 0.6,
        minTrackingConfidence: 0.6,
    });
};

window.l1StopCamera = function() {
    const c = window.l1Camera;
    c.stopped = true;
    if (c.raf !== undefined) cancelAnimationFrame(c.raf);
    if (c.stream) c.stream.getTracks().forEach(track => track.stop());
    try { if (c.tracker) c.tracker.close(); }
    finally {
        if (c.wrapper) c.wrapper.remove();
        window.l1Camera = {};
    }
};

window.l1StartCamera = async function(options = {}) {
    const captureWidth = Math.max(160, Math.min(1280, options.width || 480));
    const jpegQuality = Math.max(0.01, Math.min(1, options.quality || 0.75));
    window.l1StopCamera();
    const c = window.l1Camera;
    const active = () => window.l1Camera === c && !c.stopped;
    try {
        c.wrapper = document.createElement('div');
        c.video = document.createElement('video');
        c.video.style.display = 'none';
        c.video.muted = true;
        c.video.playsInline = true;
        c.preview = document.createElement('canvas');
        c.preview.style.maxWidth = '640px';
        c.preview.style.width = '100%';
        c.status = document.createElement('p');
        c.status.setAttribute('role', 'status');
        c.light = {x: -0.6, y: -0.4, z: 0.0, power: 4.0, specular: 0.3};
        c.gesture = new window.L3GestureControl();
        c.mode = options.hand ? 'gesture' : 'manual';
        c.tracking = options.hand ? 'Loading hand model — manual sliders available' : 'Manual control';
        const controls = document.createElement('div');
        controls.style.maxWidth = '640px';
        const modeLabel = document.createElement('label');
        const mode = document.createElement('input');
        mode.type = 'checkbox'; mode.checked = !!options.hand;
        mode.setAttribute('aria-label', 'Hand control');
        modeLabel.append(mode, ' Hand control (uncheck for manual XYZ)');
        const inputs = {};
        const syncControls = () => {
            for (const key of ['x', 'y', 'z', 'power', 'specular']) {
                const {input, caption, label} = inputs[key];
                input.value = c.light[key];
                input.disabled = ['x', 'y', 'z'].includes(key) && c.mode === 'gesture' && !!c.tracker;
                caption.textContent = label + ': ' + c.light[key].toFixed(2) + ' ';
            }
            c.status.textContent = c.tracking + ' | Light XYZ: ' +
                ['x', 'y', 'z'].map(k => c.light[k].toFixed(2)).join(', ') + ' (arbitrary units)';
        };
        const addSlider = (key, label, min, max, step) => {
            const row = document.createElement('label');
            row.style.display = 'block';
            const caption = document.createElement('span');
            const input = document.createElement('input');
            input.type = 'range'; input.min = min; input.max = max; input.step = step;
            input.style.width = '60%';
            input.setAttribute('aria-label', label);
            input.oninput = () => { c.light[key] = Number(input.value); syncControls(); };
            inputs[key] = {input, caption, label, initial: c.light[key]};
            row.append(caption, input);
            controls.append(row);
        };
        addSlider('x', 'Light X (right)', -3, 3, 0.05);
        addSlider('y', 'Light Y (down)', -2, 2, 0.05);
        addSlider('z', 'Light Z (toward scene)', -2, 0.7, 0.05);
        addSlider('power', 'Power', 0, 12, 0.1);
        addSlider('specular', 'Specular', 0, 1, 0.05);
        mode.onchange = () => {
            c.mode = mode.checked ? 'gesture' : 'manual';
            c.gesture.reset();
            c.tracking = c.mode === 'manual' ? 'Manual control' :
                (c.tracker ? 'Show one palm to calibrate Z' : 'Hand model unavailable or loading — use manual sliders');
            if (c.mode === 'gesture') ensureTracker();
            syncControls();
        };
        const calibrate = document.createElement('button');
        calibrate.textContent = 'Calibrate hand Z';
        calibrate.onclick = () => {
            c.gesture.reset();
            c.tracking = 'Calibration reset — show one palm at a comfortable distance';
            syncControls();
        };
        const reset = document.createElement('button');
        reset.textContent = 'Reset light';
        reset.onclick = () => {
            for (const [key, item] of Object.entries(inputs)) c.light[key] = item.initial;
            c.gesture.reset();
            c.tracking = c.mode === 'gesture' ? 'Light reset — show one palm to calibrate Z' : 'Manual control';
            syncControls();
        };
        const stop = document.createElement('button');
        stop.textContent = 'Stop camera'; stop.onclick = window.l1StopCamera;
        c.wrapper.append(c.video, c.preview, c.status, modeLabel, controls, calibrate, reset, stop);
        document.body.append(c.wrapper);
        syncControls();
        c.stream = await navigator.mediaDevices.getUserMedia({
            video: {width: {ideal: 640}, height: {ideal: 480}}, audio: false,
        });
        if (!active()) { c.stream.getTracks().forEach(t => t.stop()); return false; }
        c.video.srcObject = c.stream;
        await c.video.play();
        if (!active()) return false;
        if (!c.video.videoWidth || !c.video.videoHeight) throw new Error('Camera returned invalid dimensions');
        c.canvas = document.createElement('canvas');
        c.lastVideoTime = -1;
        c.lastTick = -Infinity;
        c.frameId = 0;
        c.updateFrame = now => {
            if (!active() || c.video.readyState < 2 || c.video.currentTime === c.lastVideoTime) return;
            const scale = Math.min(1, captureWidth / c.video.videoWidth);
            const width = Math.max(3, Math.round(c.video.videoWidth * scale));
            const height = Math.max(3, Math.round(c.video.videoHeight * scale));
            if (c.canvas.width !== width || c.canvas.height !== height) {
                c.canvas.width = c.preview.width = width;
                c.canvas.height = c.preview.height = height;
                c.gesture.reset();
            }
            c.canvas.getContext('2d').drawImage(c.video, 0, 0, width, height);
            c.lastVideoTime = c.video.currentTime;
            c.capturedAt = now;
            c.frameId++;
            let landmarks;
            if (c.tracker && c.mode === 'gesture') {
                try {
                    landmarks = c.tracker.detectForVideo(c.canvas, now).landmarks[0];
                    c.tracking = c.gesture.update(landmarks, width, height, now, c.light);
                } catch (error) {
                    const failed = c.tracker;
                    c.tracker = null;
                    try { failed.close(); } catch (_) {}
                    c.mode = 'manual'; mode.checked = false;
                    c.tracking = 'Hand tracking failed — manual control: ' + error.message;
                }
            }
            // Preview landmarks never enter the RGB image sent to the depth model.
            const ctx = c.preview.getContext('2d');
            ctx.drawImage(c.canvas, 0, 0);
            if (landmarks) {
                ctx.fillStyle = '#00ffb3';
                for (const point of landmarks) {
                    ctx.beginPath(); ctx.arc(point.x * width, point.y * height, 3, 0, 2 * Math.PI); ctx.fill();
                }
            }
            syncControls();
        };
        c.updateFrame(performance.now());
        const tick = now => {
            if (!active()) return;
            // At most 20 tracking updates/s; synchronous WASM can still block the UI.
            if (now - c.lastTick >= 50) { c.lastTick = now; c.updateFrame(now); }
            c.raf = requestAnimationFrame(tick);
        };
        c.raf = requestAnimationFrame(tick);
        // Camera/manual controls work while the assets load, or if loading fails.
        function ensureTracker() {
            if (c.tracker || c.loading) return;
            c.loading = window.l3LoadHandTracker().then(tracker => {
            if (!active()) { tracker.close(); return; }
            c.tracker = tracker;
            c.tracking = c.mode === 'gesture' ? 'Show one palm to calibrate Z' : 'Manual control';
            syncControls();
        }).catch(error => {
            if (!active()) return;
            c.mode = 'manual'; mode.checked = false;
            c.tracking = 'Hand model could not load — manual control: ' + error.message;
            syncControls();
            }).finally(() => { c.loading = null; });
        };
        c.jpegQuality = jpegQuality;
        if (c.mode === 'gesture') ensureTracker();
        return true;
    } catch (error) {
        if (active()) window.l1StopCamera();
        throw error;
    }
};

window.l1CaptureFrame = function() {
    const c = window.l1Camera;
    if (!c.stream || !c.canvas || !c.stream.getVideoTracks().some(t => t.readyState === 'live')) return null;
    // Manual Level 2 captures can use a fresh video frame, beyond the 20 Hz tracker cap.
    if (c.mode === 'manual') c.updateFrame(performance.now());
    // Snapshot the last tracked canvas and its control state atomically; no frame queue.
    return {
        image: c.canvas.toDataURL('image/jpeg', c.jpegQuality), light: {...c.light},
        hand: {mode: c.mode, status: c.tracking, frame_id: c.frameId,
               age_ms: Math.max(0, performance.now() - c.capturedAt)},
    };
};

    """))
    import json
    options = dict(width=CAPTURE_WIDTH, quality=JPEG_QUALITY / 100, hand=HAND_CONTROL)
    return eval_js("l1StartCamera(" + json.dumps(options) + ")")


def get_colab_frame():
    from google.colab.output import eval_js
    data = eval_js("l1CaptureFrame()")
    if not data:
        return None
    binary = b64decode(data["image"].split(",", 1)[1])
    bgr = cv2.imdecode(np.frombuffer(binary, dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError("Could not decode the webcam frame.")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), data["light"], data["hand"]


def stop_colab_webcam():
    from google.colab.output import eval_js
    eval_js("window.l1StopCamera && window.l1StopCamera()")


def live_panel(frame_rgb, z_proxy, normal_rgb, relit_rgb):
    scaled = np.clip((z_proxy - 1 / 1.2) / (1 / 0.2 - 1 / 1.2), 0, 1)
    depth_rgb = cv2.cvtColor(
        cv2.applyColorMap((scaled * 255).astype(np.uint8), cv2.COLORMAP_MAGMA),
        cv2.COLOR_BGR2RGB,
    )
    panels = []
    for picture, title in zip(
        (frame_rgb, depth_rgb, normal_rgb, relit_rgb),
        ("Processed RGB", "Z proxy (arbitrary units)", "Approximate normals", "Level 3: hand-controlled light"),
    ):
        panel = cv2.copyMakeBorder(picture, 32, 0, 0, 0, cv2.BORDER_CONSTANT)
        cv2.putText(panel, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        panels.append(panel)
    top = np.concatenate(panels[:2], axis=1)
    bottom = np.concatenate(panels[2:], axis=1)
    return Image.fromarray(np.concatenate((top, bottom), axis=0))


class LatestLocalCamera:
    """One latest frame, no inference backlog; camera read stays off the UI thread."""
    def __init__(self, index=0, width=480, fps=30):
        import threading
        import sys
        if width < 3 or fps <= 0:
            raise ValueError("Camera width must be >=3 and FPS positive.")
        backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
        self.capture = cv2.VideoCapture(index, backend)
        if not self.capture.isOpened():
            self.capture.release()
            raise RuntimeError(f"Could not open camera {index}; close other camera apps or change CAMERA_INDEX.")
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, round(width * 0.75))
        self.capture.set(cv2.CAP_PROP_FPS, fps)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Best effort; some drivers ignore it.
        self.width = width
        self.condition = threading.Condition()
        self.stopped = False
        self.latest = None
        self.sequence = self.delivered = 0
        self.error = None
        self.thread = threading.Thread(target=self._read, name="vision-camera", daemon=True)
        self.thread.start()

    def _read(self):
        try:
            while not self.stopped:
                ok, bgr = self.capture.read()
                if not ok or bgr is None:
                    raise RuntimeError("Camera stopped returning frames.")
                captured_at = time.perf_counter()
                h, w = bgr.shape[:2]
                if w > self.width:
                    bgr = cv2.resize(bgr, (self.width, max(3, round(h*self.width/w))),
                                     interpolation=cv2.INTER_AREA)
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                with self.condition:
                    if self.stopped:
                        break
                    self.sequence += 1
                    self.latest = (rgb, captured_at, self.sequence)
                    self.condition.notify_all()
        except Exception as error:
            with self.condition:
                self.error = error
                self.condition.notify_all()
        finally:
            self.capture.release()

    def get(self, timeout=5.0):
        with self.condition:
            ready = self.condition.wait_for(
                lambda: self.stopped or self.error is not None or self.sequence > self.delivered,
                timeout=timeout,
            )
            if self.stopped:
                return None
            if self.error is not None:
                raise self.error
            if not ready:
                raise RuntimeError("Timed out waiting for a new camera frame.")
            rgb, captured_at, sequence = self.latest
            self.delivered = sequence
            return rgb, {"mode": "manual", "status": "Local XYZ sliders", "frame_id": sequence,
                         "age_ms": max(0, (time.perf_counter()-captured_at)*1000)}

    def close(self):
        with self.condition:
            self.stopped = True
            self.condition.notify_all()
        self.thread.join(timeout=1.0)
        if self.thread.is_alive():
            self.capture.release()  # Unblock drivers that support cancelling read.
            self.thread.join(timeout=1.0)
        if self.thread.is_alive():
            raise RuntimeError("Camera driver did not stop; restart the kernel before reopening the camera.")


def is_local_backend():
    backend = globals().get("LIVE_BACKEND", "local")
    if backend not in ("local", "colab"):
        raise ValueError('LIVE_BACKEND must be "local" or "colab".')
    return backend == "local"


def local_light_specs():
    return [("x", "X", -3., 3., 0.05), ("y", "Y", -2., 2., 0.05),
            ("z", "Z", -2., 0.7, 0.05), ("power", "Power", 0., 12., 0.1),
            ("specular", "Specular", 0., 1., 0.05)]


def reset_local_light():
    values = default_light()
    for key, label, low, high, step in local_light_specs():
        cv2.setTrackbarPos(label, _local_window, round((values[key]-low)/step))


def start_webcam():
    if not is_local_backend():
        return start_colab_webcam()
    global _local_camera, _local_window
    old = globals().get("_local_camera")
    if old is not None:
        old.close()
    _local_camera = None
    _local_window = "AI Vision - relighting (Q/Esc: stop, R: reset)"
    try:
        _local_camera = LatestLocalCamera(CAMERA_INDEX, CAPTURE_WIDTH, CAMERA_FPS)
        cv2.namedWindow(_local_window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(_local_window, max(640, CAPTURE_WIDTH), 540)
        for key, label, low, high, step in local_light_specs():
            cv2.createTrackbar(label, _local_window, 0, round((high-low)/step), lambda _: None)
        reset_local_light()
        return True
    except Exception:
        stop_webcam()
        raise


def local_window_open():
    try:
        return cv2.getWindowProperty(_local_window, cv2.WND_PROP_VISIBLE) >= 1
    except cv2.error:
        return False


def get_frame():
    if not is_local_backend():
        return get_colab_frame()
    if not local_window_open():
        return None
    frame = _local_camera.get()
    if frame is None:
        return None
    rgb, hand = frame
    light = {key: low + cv2.getTrackbarPos(label, _local_window)*step
             for key, label, low, high, step in local_light_specs()}
    return rgb, light, hand


def show_local_frame(rgb):
    if not local_window_open():
        return False
    cv2.imshow(_local_window, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    key = cv2.waitKey(1) & 0xFF
    if key in (ord("q"), 27):
        return False
    if key == ord("r"):
        reset_local_light()
    return local_window_open()


def stop_webcam():
    if not is_local_backend():
        return stop_colab_webcam()
    global _local_camera
    camera = globals().get("_local_camera")
    try:
        if camera is not None:
            camera.close()
    finally:
        # Retain an unclosed camera reference so a rerun retries its cleanup.
        if camera is None or not camera.thread.is_alive():
            _local_camera = None
        try:
            cv2.destroyWindow(globals().get("_local_window", "AI Vision"))
            cv2.waitKey(1)
        except cv2.error:
            pass


class StageTimings:
    """CPU wall timings + CUDA event intervals without per-stage barriers.

    GPU event times include stream idle time between events (e.g. preprocessing).
    Display timing is encoding/dispatch, not network delivery or browser paint.
    Resolve only after the final blocking image download completes.
    """
    def __init__(self, device, enabled=True):
        self.cuda = enabled and torch.device(device).type == "cuda"
        self.enabled = enabled
        self.events, self.starts, self.ms = {}, {}, {}

    def begin(self, name):
        if not self.enabled:
            return
        self.starts[name] = time.perf_counter()
        if self.cuda:
            pair = self.events.setdefault(name, (torch.cuda.Event(enable_timing=True),
                                                  torch.cuda.Event(enable_timing=True)))
            pair[0].record()

    def end(self, name):
        if not self.enabled:
            return
        self.ms[name] = (time.perf_counter() - self.starts[name]) * 1000
        if self.cuda:
            self.events[name][1].record()

    def resolve(self):
        return {name: a.elapsed_time(b) for name, (a, b) in self.events.items()} if self.cuda else dict(self.ms)


def encode_live_image(rgb):
    # Explicit JPEG avoids IPython's implicit PNG encoding of PIL objects.
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        raise RuntimeError("Could not encode the rendered frame.")
    return DisplayImage(data=encoded.tobytes(), format="jpeg")


def compute_live_geometry(raw, mapper, estimator):
    if isinstance(mapper, DeviceRelativeGeometryDepth):
        depth = mapper(raw, temporal=True, validate=False)
    else:
        depth = mapper(raw.cpu().numpy(), temporal=True)
    depth = torch.as_tensor(depth, device=estimator.device, dtype=torch.float32)
    normals, points, valid = estimator.compute_normals_and_coords(depth, validate=False)
    return depth, normals, points, valid


def select_geometry_mapper(raw, estimator):
    """Benchmark complete normalization+normals paths once per frame shape.

    CPU OpenCV can beat GPU tensor smoothing on smaller GPUs. Warmup and
    synchronization are confined to startup; steady-state frames have no
    calibration barriers. Return a fresh mapper so profiling cannot alter EMA.
    """
    choice = globals().get("GEOMETRY_BACKEND", "auto")
    factories = {"cpu": RelativeGeometryDepth, "gpu": DeviceRelativeGeometryDepth}
    if choice not in ("auto", "cpu", "gpu"):
        raise ValueError('GEOMETRY_BACKEND must be "auto", "cpu", or "gpu".')
    if choice != "auto":
        return factories[choice](), choice
    if raw.device.type != "cuda":
        return RelativeGeometryDepth(), "cpu"
    durations = {}
    for name, factory in factories.items():
        candidate = factory()
        for _ in range(2):
            compute_live_geometry(raw, candidate, estimator)
        torch.cuda.synchronize(raw.device)
        start = time.perf_counter()
        for _ in range(5):
            compute_live_geometry(raw, candidate, estimator)
        torch.cuda.synchronize(raw.device)
        durations[name] = (time.perf_counter()-start)*1000/5
    selected = min(durations, key=durations.get)
    print(f"Geometry auto: CPU {durations['cpu']:.1f} ms, GPU {durations['gpu']:.1f} ms; using {selected.upper()} smoothing.")
    return factories[selected](), selected


def run_live():
    status = display(Markdown("Starting camera..."), display_id=True)
    output = None if is_local_backend() else display(DisplayImage(data=b"", format="jpeg"), display_id=True)
    mapper, geometry_backend = None, "pending"
    estimator = None
    cycles, stage_samples = deque(maxlen=60), deque(maxlen=60)
    timer = StageTimings(DEVICE, PROFILE_STAGES)
    completed, last_status = 0, -float("inf")
    try:
        if not start_webcam():
            return
        while True:
            start = time.perf_counter()
            packet = get_frame()
            if packet is None:
                break
            capture_ms = (time.perf_counter() - start) * 1000
            frame_rgb, light, hand = packet
            h, w = frame_rgb.shape[:2]
            if estimator is None or (estimator.H, estimator.W) != (h, w):
                estimator = ScharrNormalEstimator(h, w, hfov_deg=ASSUMED_HFOV_DEG, device=DEVICE)
                mapper = None
            raw = infer_relative_inverse(frame_rgb, return_tensor=True, timings=timer)
            if mapper is None:
                mapper, geometry_backend = select_geometry_mapper(raw, estimator)
            timer.begin("geometry")
            z_proxy, normals, points, valid = compute_live_geometry(raw, mapper, estimator)
            timer.end("geometry")
            timer.begin("lighting")
            relit = relight_rgb(frame_rgb, normals, points, valid, light, return_tensor=True)
            timer.end("lighting")
            download_start = time.perf_counter()
            if SHOW_DIAGNOSTICS:
                # One packed download for all GPU-generated debug panels.
                normal_rgb = ((normals + 1) * 127.5).round().clamp(0, 255).to(torch.uint8)
                normal_rgb = torch.where(valid[..., None], normal_rgb, torch.zeros_like(normal_rgb))
                z_byte = ((z_proxy - 1/1.2) / (1/0.2 - 1/1.2) * 255).clamp(0, 255).to(torch.uint8)
                packed = torch.cat((relit, normal_rgb, z_byte[..., None]), -1).cpu().numpy()
                # live_panel accepts a float proxy; reconstruct only for its colormap.
                debug_z = packed[..., 6].astype(np.float32) / 255 * (1/0.2 - 1/1.2) + 1/1.2
                picture = np.asarray(live_panel(frame_rgb, debug_z, packed[..., 3:6], packed[..., :3]))
            else:
                picture = relit.cpu().numpy()
            download_ms = (time.perf_counter() - download_start) * 1000
            stage_ms = timer.resolve()  # Last .cpu() has already completed the CUDA events.
            display_start = time.perf_counter()
            if is_local_backend():
                picture = picture.copy()
                rate_label = f"{len(cycles)/sum(cycles):.1f} FPS" if cycles else "Warming up"
                cv2.putText(picture, f"{rate_label} | {w}x{h} | Q: stop, R: reset",
                            (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 160), 1, cv2.LINE_AA)
                if not show_local_frame(picture):
                    break
            else:
                output.update(encode_live_image(picture))
            stage_ms.update(capture=capture_ms, download_wait=download_ms,
                            display=(time.perf_counter() - display_start) * 1000)
            now = time.perf_counter()
            if now - last_status >= STATUS_INTERVAL:
                last_status = now
                fps = len(cycles) / sum(cycles) if cycles else None
                rate = f"{fps:.2f} completed cycles/s" if fps is not None else "warming up"
                if stage_samples:
                    keys = ("capture", "preprocess", "depth", "geometry", "lighting", "download_wait", "display")
                    timing_text = " | ".join(f"{key}: {np.mean([s[key] for s in stage_samples if key in s]):.1f} ms"
                                             for key in keys if any(key in s for s in stage_samples))
                else:
                    timing_text = "Collecting stage timings after warmup"
                status.update(Markdown(
                    f"**Loop rate:** {rate} (browser paint excluded)  \n"
                    f"**Stages ({geometry_backend.upper()} smoothing):** {timing_text}  \n"
                    "CUDA stages overlap CPU work; download_wait includes queued GPU work. Do not add these timings.  \n"
                    f"**Light XYZ:** ({light['x']:+.2f}, {light['y']:+.2f}, {light['z']:+.2f}) arbitrary units | "
                    f"power: {light['power']:.1f} | specular: {light['specular']:.2f}  \n"
                    f"**Hand control:** {hand['mode']} | {hand['status']} | capture age: {hand['age_ms']:.0f} ms  \n"
                    + ("Use the OpenCV window sliders for XYZ, power and specular. Press **Q/Esc** or close the window to stop."
                     if is_local_backend() else
                     "Use XYZ sliders for Level 2; enable **Hand control** for Level 3. **Stop camera** to finish.")
                ))
            completed += 1
            if completed > 3:
                cycles.append(time.perf_counter() - start)
                stage_samples.append(stage_ms)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            stop_webcam()
        except Exception as cleanup_error:
            print(f"Camera cleanup could not be confirmed: {cleanup_error}")
        rate = f" Last measured loop rate: {len(cycles) / sum(cycles):.2f} cycles/s." if cycles else ""
        status.update(Markdown(f"**Processing stopped.**{rate}"))


run_live()

