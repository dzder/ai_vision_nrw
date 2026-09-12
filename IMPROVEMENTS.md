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
