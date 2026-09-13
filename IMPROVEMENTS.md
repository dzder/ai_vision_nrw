# Real-time performance and visual quality backlog

Recorded: September 12, 2026.

Active implementation: [test_l1.ipynb](test_l1.ipynb).
Architecture and completed work: [context.md](context.md).

This backlog records the user's review and work still needed. The reported
**2.9 FPS**, hand jitter, and weak shadow appearance need a reproducible local
measurement. This review does not establish which stage causes the slowdown.
All unchecked items below are future work, not completed fixes.

## Targets and priorities

- **P0 — performance:** reproduce 2.9 FPS, identify the bottleneck, and improve
  actual completed-loop throughput. Aim for sustained **20–30+ FPS** with depth,
  normals, hand tracking, lighting, and shadows enabled. This is our desired
  target, not a verified numerical competition requirement.
- **P1 — stable geometry and control:** reduce unintended light movement, make
  relative Z control obvious, and preserve hand/face/background boundaries.
- **P1 — shadow quality:** produce coherent cast shadows that follow light and
  subject movement without excessive self-shadowing, detached edges, or trails.
- **P2 — presentation:** improve perceived lighting realism and provide clear,
  truthful demo feedback. Basic diagnostic telemetry belongs in P0.

## 1. Diagnose and improve performance — P0

**Reported problem:** approximately 2.9 FPS, making interaction visibly delayed.

**Already implemented:** latest-frame camera capture, CUDA fp16 depth inference,
GPU-capable normals/rendering/shadows, optional CPU/GPU geometry-path selection,
and stage timing. The shadow tracer uses batched PyTorch operations; CPU
per-pixel shadow processing has not been established as the cause.

- [ ] Reproduce from a fresh kernel with recorded device, GPU, model target,
  actual input dimensions, capture size, shadow preset, and diagnostic settings.
  Confirm updated helper cells are loaded before starting live processing.
- [ ] Log completed-loop FPS and per-frame processing time after warmup; report
  mean and p95 over a sustained run, including stalls and dropped capture frames.
- [ ] Inspect capture, hand, preprocessing, depth, geometry, shadow, lighting,
  download-wait, and display timings. CUDA/CPU stages overlap; do not add their
  reported durations together or treat queued GPU wait as an independent cost.
- [ ] Compare the same scene with hand tracking off/on, shadows off/on, fast
  versus quality shadows, diagnostics off/on, and each camera/model preset.
- [ ] Check CPU fallback, GPU contention, repeated transfers, unnecessary
  synchronization, stale camera buffering, and window/status rendering overhead.
- [ ] Optimize the measured bottleneck first. Consider smaller hand input,
  bounded asynchronous processing, or further inference optimization only after
  measuring their effect on alignment, quality, latency, and throughput.
- [ ] Retain frame IDs and timestamps if concurrency changes. Every displayed
  processed frame must retain its own depth/geometry and associated light state;
  do not inflate FPS by redisplaying old results or silently freezing geometry.
- [ ] Re-benchmark on the intended demo machine and record the configuration
  that meets the best practical speed/quality balance.

**Known constraint:** the saved MX550 experiment measured roughly 57 ms for
PyTorch depth alone at its fast configuration, before hand tracking and display.
That configuration cannot reach 20 FPS in a serial loop without further changes.
The ONNX experiment was slower in that comparison, so switching runtimes is not
an established fix. The later shadow benchmark measured **8.29 ms mean added
rendering compute at 320×240**, not total live processing time.

**Acceptance:** publish a reproducible live result with all required stages
enabled. If 20–30 FPS remains infeasible, identify the measured limiting stage
and the necessary model, quality, or hardware tradeoff; do not claim the target.

## 2. Stabilize hand tracking and light control — P1

**Reported problem:** noisy finger landmarks and unintended light/shadow jumps.

**Already implemented:** a time-based 120 ms EMA on light XYZ, a 180 ms EMA on
power, acquisition gating, loss handling, and manual fallback. Raw landmark
overlay jitter and actual light-coordinate jitter are different measurements.

- [ ] Measure raw palm position/span, filtered light XYZ, and power while the
  hand is stationary, moving slowly, moving quickly, and partially occluded.
- [ ] Reject implausible landmark jumps and unreliable palm measurements before
  mapping them to the light; keep loss/reacquisition behavior explicit.
- [ ] Compare the existing EMA against an adaptive filter such as One-Euro.
  Tune for low stationary jitter without excessive response lag; avoid stacking
  filters blindly. Evaluate a Kalman filter only if it offers a measured benefit.
- [ ] Smooth displayed landmarks separately if needed; do not feed overlay
  pixels back into inference or make a smoothed overlay conceal unstable control.
- [ ] Verify stable intensity while fingers open/close and while the palm rotates.

**Acceptance:** a stationary hand produces stable light coordinates and shadows;
intentional motion remains responsive, with no large jump on reacquisition.

## 3. Handle hand/face occlusion and depth discontinuities — P1

**Reported problem:** a hand near the camera covers the face, introducing thin
fingers, large depth differences, and self-occlusion.

- [ ] Establish repeatable scenes: hand beside the face, hand over the face,
  separated fingers, overlapping fingers, and foreground against a distant wall.
- [ ] Inspect depth and validity around boundaries to distinguish model errors
  from gradient, shadow sampling, and filtering artifacts.
- [ ] Prevent smoothing and derivative neighborhoods from mixing unrelated
  surfaces across large depth jumps. Treat occluded geometry as unknown.
- [ ] Preserve thin structures where the model resolves them; identify detail
  that the low-resolution depth prediction cannot recover.
- [ ] Add demo guidance for a comfortable hand position beside the face, while
  still testing the difficult occlusion cases rather than hiding them.

**Acceptance:** crossing foreground/background boundaries does not create broad
normal halos or spurious bridges between the hand, face, and background.

## 4. Improve surface normals — P1

**Reported problem:** noisy or pixelated hand/face shading and unclear normal
continuity within surfaces. Real object boundaries should retain discontinuities.

**Already implemented:** a positive relative-Z proxy, edge-preserving bilateral
filtering, perspective Scharr tangents, unit normals, and a validity mask. The
depth comes from a monocular model, not a physical depth sensor.

- [ ] Inspect synchronized RGB, Z, normals, and relighting to locate the source
  of artifacts rather than judging normals from the relit image alone.
- [ ] Detect invalid values or actual holes; fill only supported small gaps
  within the same surface, without inventing hidden/background geometry.
- [ ] Compare discontinuity-aware tangent estimation with the current Scharr
  neighborhoods; prevent derivatives from spanning unrelated surfaces.
- [ ] Evaluate depth- and normal-guided smoothing within surfaces, renormalizing
  vectors afterward and preserving validity and boundary masks.
- [ ] Compare depth model resolution settings against their measured FPS cost.
- [ ] Add tilted-plane, thin-foreground, and depth-step regression scenes.

**Acceptance:** smooth, unit-length normals on continuous surfaces; sharp object
boundaries; fewer flat-surface ripples and boundary halos without unacceptable
loss of finger detail or throughput.

## 5. Improve perceived lighting realism — P2

**Reported problem:** relighting looks like a visual effect rather than a light
occupying a consistent position in the reconstructed scene.

**Already implemented:** geometry-driven Lambertian diffuse, Blinn–Phong
specular, softened distance attenuation, ambient light, and linear-RGB shading.
These terms need evaluation/tuning, not duplicate implementation.

- [ ] Verify coordinate conventions and light-direction consistency using a
  simple plane and foreground object before tuning a complex face/hand scene.
- [ ] Tune power, ambient, specular strength, and shininess to avoid washed-out
  highlights and overly dark surfaces; compare identical camera exposure settings.
- [ ] Validate changing light distance separately from changing hand-controlled
  intensity, so the attenuation effect is observable.
- [ ] Account for camera auto-exposure changes when judging temporal brightness.
- [ ] Record the limitations of using already-lit RGB as approximate albedo and
  relative depth as geometry. True removal of existing illumination would need
  a separate approach and performance budget.

**Acceptance:** light motion produces consistent diffuse response, highlight
movement, attenuation, and cast-shadow direction without widespread clipping.

## 6. Improve cast-shadow quality — P1

**Reported problem:** shadows appear insufficiently clean or physically defined.

**Already implemented:** receiver-to-light tracing through the visible depth
layer, perspective-correct depth interpolation, finite occluder thickness,
receiver bias, and depth-aware filtering/upsampling. Shadows attenuate direct
diffuse/specular while preserving ambient illumination.

- [ ] Use the visibility view (V) to distinguish actual occlusion errors from
  normal, albedo, or lighting artifacts.
- [ ] Measure fast versus quality tracing before selecting a default. More
  samples must be justified by visible improvement and the live performance budget.
- [ ] Refine sampling near silhouettes and uncertain hits to reduce gaps and
  missed thin occluders without increasing work uniformly across the image.
- [ ] Evaluate surface-angle/pixel-size-aware bias and thickness to balance
  self-shadowing, shadow detachment, and light leakage.
- [ ] Preserve boundaries when filtering and upsampling visibility.
- [ ] Add finite-light-size softness if performance permits: sharp contact
  shadows and broader penumbrae with greater blocker/receiver separation.
  A larger uniform blur alone does not provide this behavior.
- [ ] If temporal shadow filtering is introduced, reproject history and reject
  stale samples after motion, disocclusion, light changes, or resolution changes.
- [ ] Validate moving light and moving subjects against a visible background
  receiver; preserve analytic geometry and CPU/CUDA regression checks.

**Acceptance:** shadows originate consistently from foreground subjects, move
appropriately with XYZ light motion, and avoid large false dark patches or trails.
Hidden and off-camera occluders remain a limitation of the current depth layer.

## 7. Make relative light Z-control robust and obvious — P1

**Reported problem:** it is difficult to tell whether approach/retreat reliably
changes the light's depth.

**Already implemented:** calibrated palm-span-to-Z mapping, smoothing, and the C
recalibration key. This is relative control in arbitrary units, not recovered
metric hand distance; palm rotation can imitate distance changes.

- [ ] Test approach/retreat with a steady palm orientation and separately test
  rotation, finger movement, partial clipping, and loss/reacquisition.
- [ ] Compare palm-size estimators using multiple reliable landmarks; reject
  distorted or unreliable measurements instead of updating Z abruptly.
- [ ] Tune Z sensitivity, smoothing, deadband, and bounds against measured
  stationary jitter and intentional motion. Retain explicit recalibration.
- [ ] Display live Z and calibration/tracking status; add a simple depth indicator
  that makes approach/retreat understandable without reading arbitrary units.
- [ ] Verify: hand approaches camera → light Z decreases; hand retreats → light
  Z increases, with consistent lighting and shadow changes for the test scene.

**Acceptance:** deliberate approach/retreat has a clear, repeatable visual effect
with little drift while holding still and no major jump after recalibration/loss.

## 8. Add clear, truthful demo feedback — P0 diagnostics / P2 presentation

- [ ] Show completed-loop FPS, actual execution device, and shadow quality/state.
- [ ] Show depth, normals, and hand states derived from successful processing of
  the current frame. Distinguish OK, invalid/degraded, lost, disabled, and error.
- [ ] Show light X/Y/Z, power, hand/manual mode, and Z-calibration status.
- [ ] Add measured frame age at display submission and processing time. Label
  them precisely: neither automatically measures sensor-to-screen latency.
- [ ] Keep detailed stage profiling optional and updates throttled so the HUD
  does not become a significant source of slowdown.
- [ ] Keep text readable at 320-pixel capture width, preserve access to diagnostic
  views, and show the main controls without covering important scene content.

Suggested fields (populate with actual values, never demonstration numbers):

```text
FPS: <completed loops/s>        Device: <actual device>
Depth: <state>  Normals: <state>  Hand: <state>
Light: X <value>  Y <value>  Z <value> [relative units]
Power: <value>  Shadows: <on/off> / <quality>
Frame age at display submission: <measured ms>
Processing: <measured ms>       Mode: <hand/manual>
```

## Recommended execution order and evidence

1. Reproduce the slow live run and add reliable profiling/feedback.
2. Remove the measured performance bottleneck and establish a live baseline.
3. Stabilize palm/XYZ/power control and depth boundaries; improve normals.
4. Refine shadow sampling/contact behavior, then evaluate additional softness.
5. Tune lighting and polish the HUD; repeat the full live acceptance sequence.

- [ ] Save before/after configurations, live FPS and p95 processing times, stage
  measurements, and synchronized diagnostic images for each material change.
- [ ] Test stationary and moving subjects, XYZ motion, intensity, hand loss and
  reacquisition, manual fallback, camera failure, and quit/restart cleanup.
- [ ] Keep recorded-demo and live-demo capabilities consistent. Synthetic tests
  and compute-only benchmarks supplement but do not replace live validation.
- [ ] Update [context.md](context.md) with measured outcomes and unresolved issues.
- [ ] Track the later notebook-to-`python main.py` submission integration
  separately; the active improvement work remains in the notebook.

## Completed: hand-centered shading cull + flat-surface suppression — September 13, 2026

Notebook `test_l1.ipynb` cell-6 `relight_rgb` and desktop `lighting.shade()` now
share the same two shading fixes; both are wired on by default.

**Hand-centered shading cull** (`cull_radius_frac`, default 0.35).
`_light_screen_pixels` inverts the hand mapping (x = 6*(u-0.5), y = 4*(v-0.5))
to center a circle on the light's screen position. Per-pixel diffuse/specular
math runs only inside the bounding box: points, normals, validity and albedo are
**sliced before compute** (crop-and-slice, not a post-hoc mask) and the result is
pasted onto a full-frame ambient base. Outside the box every pixel is
`decode(albedo) * ambient` re-encoded — never the raw camera pixels. The
crop-and-slice claim is verified by a radius-sweep timing check in
`test_notebook_lighting_cull.benchmark`, not just by visual diff: on CPU the
measured relight time scales with the box area (radius 0.0 → 0.35 → 0.15 gives
46.81 → 34.85 → 18.55 ms at 480x360 torch-CPU; the numpy `shade` path is
33.32 → 24.59 → 13.33 ms). A mask-after implementation would keep timing flat.

**Flat-surface suppression** (`flat_threshold` default 0.95, `flat_suppress`
default 0.25). Raw normals with Nz above the threshold (background walls) keep
ambient but scale direct diffuse/specular by `flat_suppress`. The bump-slope
pixels (hand-sized) are byte-for-byte identical to the baseline; ambient
(power=0) renders are unchanged.

Honest per-frame numbers on the GTX 1660 Ti (synthetic, not live FPS):
depth inference alone is **40.69 ms @196, 43.62 ms @224, 75.96 ms @294**
(`validation/depth_only_benchmark.json`) → a serial loop cannot exceed
**~24.6 / 22.9 / 13.2 FPS** before geometry, hand, shadading, shadows, download
or display. On this GPU the shaded stage is launch-bound: cull is neutral
(±0.1-0.9 ms) at 320-640 px widths because the per-pixel GPU math it removes is
already cheap (`validation/lighting_cull/benchmark.json`). The cull saves real
time on the CPU/numpy desktop path (see sweep above) and wherever per-pixel
costs dominate (weak GPUs, larger frames, threaded pipeline). Depth, not
shading, is the current bottleneck: further lighting micro-tuning yields
single-digit percent frame-time gains until the depth stage streams ahead of the
render stages.

**Notebook-to-`main.py` integration (submission path).** Ported the same
`cull_radius_frac` / `flat_threshold` / `flat_suppress` semantics into
`lighting.shade()` (numpy, orthographic view, same scale mapping as
`LightController`). `python main.py` accepts `--cull-radius-frac`,
`--flat-threshold`, `--flat-suppress`, draws the ambient-only boundary circle on
the relit panel, and reports cull/flat state in the status lines. `main.py`
remains Level 2 (no hand tracking/shadows); Levels 3-4 stay in the notebook for
now.

**Process note.** A transient `KeyError: relight_rgb` in the new test was
root-caused in one pass: `load_helpers()` returns `(nb, ns)`, and the new test
unpacked `ns, _ = load_helpers()` — so `ns` was the notebook structure, not the
helpers namespace. Not a library or numpy/torch issue; a four-line unpacking fix
in the test. No caching/reload system was added.

## Shading cull + ambient retune — September 13, 2026 (live fix)

**Observed live bug:** with the CUDA path at default `cull_radius_frac=0.35`,
the local window showed a flat, uniformly darkened frame with a hard square
around the light — not a green color swap, not a degenerate cull box (the box
covered 61% of the frame), and not a light-fallback bug. The visible square is
the crop-and-slice boundary: outside it every pixel is `albedo * ambient`
re-encoded. With `ambient=0.18`, ambient-only regions crushed to mean **~54**
against **~115** inside the box (61-step jump) and the whole frame dropped
**134.8 → 91.6** (mean). Root cause measured with source-injected
instrumentation, not inspection: on a CUDA path where relight is launch-bound,
the cull costs nothing (±0.1-0.9 ms) but the box's brightness discontinuity is
always visible.

**Fix (both notebook and desktop, wired on by default):**
1. `cull_radius_frac` defaults to **0 on CUDA** (evidence: relight is
   launch-bound there, full-frame is no slower and has no boundary artifact);
   CPU keeps 0.35 where the crop actually saves time (sweep above).
   Notebook cell 2: `CULL_SHADING_RADIUS_FRAC = 0.0 if DEVICE.type == "cuda"
   else 0.35`. Desktop `main.py`: `--cull-radius-frac` now defaults to auto
   (`None` → 0 on CUDA, 0.35 on CPU), resolved after the estimator is created
   and used for the shade call, the boundary-ring draw, and the status line.
2. `ambient` default raised **0.18 → 0.38** in notebook cell-6 `relight_rgb`
   and `lighting.shade()`. Measured on the exact diagnosis frame
   (`validation/debug_frame_before/frame_capture_000.png`, same no-palm
   default light, cull=0): frame-mean drop **42.3 → 7.6** (134.8 → 127.3),
   pixels <40 from 1.75% → 0.00%, whole-frame `2G/(R+B)` 0.972 → 0.976 (no
   cast). On a second, harder live frame with 4.3% cast shadows: drop
   **52.8 → 24.4**, remaining drop is legitimate Lambert shading. I stayed at
   0.38 rather than pushing toward 0.5 to keep shadow contrast; the value is a
   single knob in cell 2 / `lighting.shade`.

**Live re-verification (`validation/debug_frame/`):** the instrumented 48-frame
camera loop now prints `in_box_frac=1.0000` on every relight (full-frame, no
box). During the run a hand was detected and moved the light to (1.05, 0.17)
with open-hand power through ~8.0, confirming hand tracking still drives the
light. Before/after PNG pairs in `validation/debug_frame_before` /
`validation/debug_frame`.

**Regression re-checks:** `test_notebook_lighting_cull.py` passes on CUDA (bit
-exact cull/flat features incl. the full-frame/legacy path), `test_lighting.py`
passes (7/7, fake estimator's string `.device` handled),
`test_notebook_geometry/shadows/gestures/local/performance` and the Node
browser-controls checks all pass. `test_notebook_onnx.py` remains
environment-blocked (onnxruntime-gpu not installed) and is unrelated.
Feathering (soft cull edge) was explicitly deferred — it only matters on the
CPU crop path. Still pending: live shadow-following with an actual hand, and
the threaded capture→depth producer so the 40 ms depth stage streams ahead of
render.
