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

window.l1StartCamera = async function() {
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
        c.mode = 'gesture';
        c.tracking = 'Loading hand model — manual sliders available';
        const controls = document.createElement('div');
        controls.style.maxWidth = '640px';
        const modeLabel = document.createElement('label');
        const mode = document.createElement('input');
        mode.type = 'checkbox'; mode.checked = true;
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
            const scale = Math.min(1, 640 / c.video.videoWidth);
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
        });
        return true;
    } catch (error) {
        if (active()) window.l1StopCamera();
        throw error;
    }
};

window.l1CaptureFrame = function() {
    const c = window.l1Camera;
    if (!c.stream || !c.canvas || !c.stream.getVideoTracks().some(t => t.readyState === 'live')) return null;
    // Snapshot the last tracked canvas and its control state atomically; no frame queue.
    return {
        image: c.canvas.toDataURL('image/jpeg', 0.85), light: {...c.light},
        hand: {mode: c.mode, status: c.tracking, frame_id: c.frameId,
               age_ms: Math.max(0, performance.now() - c.capturedAt)},
    };
};
