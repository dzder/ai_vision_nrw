# NRW 8th Edition — AI & Vision Challenge — Project Context

Organized by IEEE INSAT Student Branch and IEEE RAS INSAT Chapter under
National Robotics Week's 8th edition.

## Mission

Build a real-time computer vision system that takes a standard camera
feed, extracts pixel-accurate 3D depth geometry, and renders a
user-controlled virtual light source that dynamically illuminates the
scene, casts realistic dynamic shadows, and reacts to real-time hand
gestures moving the light through 3D space (X, Y, Z).

## Level ladder

| Level | Name | Objective | Visual benchmark |
|---|---|---|---|
| 01 | Geometry engine | Extract continuous 3D surface normal vectors N(x,y) from depth gradients in real time | Live depth-to-normal map synced with the camera feed |
| 02 | Dynamic relighting | Real-time Lambertian diffuse + specular shading on the computed geometry | Realistic illumination responsive to an interactive virtual point light |
| 03 | Spatial gesture control | Real-time hand tracking controls light placement | Smooth X/Y/Z control — light moves closer/farther and side to side |
| 04 | Dynamic shadows | Geometry-aware occlusion shadows cast behind foreground subjects | Shadow vectors that adjust smoothly as gestures move the light |
| 05 | Multi-light & volumetric scattering | Multi-hand tracking drives multiple independent lights | Overlapping shadow paths + volumetric atmospheric haze |
| ∞ | Hyper-scale horizon | Open-ended — creativity is explicitly scored beyond the fixed ladder | — |

## Deliverables (all three required)

1. **Source code repo** — must launch via `python main.py`; includes
   `requirements.txt` or a `Dockerfile`.
2. **Technical architecture brief** — model selection, normal-extraction
   formulas, temporal filtering strategy, frame-rate optimization tricks.
3. **Pitch deck** — pipeline architecture, technical trade-offs, FPS
   benchmark results, live demo configuration.

## Evaluation

- **Phase 1 — pre-selection**: video + code submitted, screened on
  rendering quality, gesture responsiveness, shading/shadow precision,
  and framerate fluidity → top teams shortlisted as finalists.
- **Phase 2 — live stage defense**: jury physically tests gesture
  control on stage; team presents the technical pipeline.
  **Anti-cheat**: the live demo must match the pre-selection video
  exactly — pre-rendered footage, hardcoded depth maps, or unhandled
  frame drops during live interaction are immediate disqualification.

## Submission

- Email: `nrw8challenges@gmail.com`
- Subject line format: `[Challenge Name] – [Team Name]`

## Tech decisions so far

- Depth model: **Depth Anything V2 Small**
  (`depth-anything/Depth-Anything-V2-Small-hf`).
- Normals: Sobel/Scharr gradient of the depth map →
  normal ∝ (-∂D/∂x, -∂D/∂y, 1), normalized.
- Codebase split: `capture.py` (webcam), `depth.py` (depth model),
  `normals.py` (gradient → normal, normal → RGB), `main.py` (loop, CLI
  args, display, FPS counter).
- Building with GPT-6 Astra as the primary coding assistant.

## Progress — Level 01 (Geometry engine)

- Initial working version built across all four modules. Reviewed for
  correctness: normal math, RGB/BGR conversions, and array shapes all
  check out — no functional bugs found.
- First live test: **4.6 FPS on CUDA** — far below real-time, and
  framerate fluidity is an explicit Phase 1 grading criterion, with
  unhandled frame drops on stage triggering disqualification.
- **Root cause identified**: `depth.py` used the high-level
  `transformers.pipeline()` API — per-frame PIL round-trips, a redundant
  CPU-side resize (the pipeline already interpolates depth back to
  input size internally before returning it), and fp32 inference (no
  fp16).
- **Fix delivered, not yet benchmarked**: rewrote `depth.py` to call
  `AutoImageProcessor` + `AutoModelForDepthEstimation` directly, run fp16
  on CUDA, and upsample on-device with `torch.nn.functional.interpolate`
  instead of PIL. Same public interface (`DepthEstimator(device)` /
  `.infer(frame_bgr)`), so it's a drop-in replacement — no changes
  needed in `main.py` or `normals.py`.
- Flagged for later: `main.py`'s capture → infer → normals → render loop
  is fully serial. Not the main bottleneck yet, but worth splitting into
  producer/consumer threads once Level 03 adds a second inference pass
  (hand tracking) competing for the same thread.
- Flagged, not a bug: faint ripple/interference pattern visible on flat
  surfaces in the normal map — a known ViT patch-embedding artifact in
  Depth Anything-family models. Worth trying a larger Gaussian kernel
  (e.g. 9×9) or `cv2.bilateralFilter` in `normals.py` to smooth flat
  regions while preserving real depth edges.

## Open decision — demo machine

Weighing a move to a different PC with a better GPU. Two things to
resolve before committing:

1. Try the software fixes above first — they may close most of the gap,
   and the result will show whether the bottleneck was really GPU
   compute or pipeline waste.
2. Whichever machine is used to record the Phase 1 video must match the
   Phase 2 live demo machine (or at least its performance class) — the
   anti-cheat rule disqualifies any mismatch between recorded and live
   capability. CUDA/driver/PyTorch-build compatibility should be
   verified on any new machine early, not the night before the deadline.

## Next steps

- Re-benchmark FPS with the corrected `depth.py`.
- Decide on and lock in the demo machine.
- Move on to Level 02 (dynamic relighting) once Level 01 is real-time.

## Review update — September 12, 2026

- Added `--input-size` (default 518; try 294 or 392) to control actual model
  preprocessing resolution. Lower settings trade fine geometry for less work.
- Added stage timing and `--benchmark-frames` for repeatable FPS comparisons.
  Camera and normals still come from the same processed frame.
- Added `probe.py`: click a textured point in either panel to track it and
  inspect its unit normal, direction change, and per-frame relative depth.
  A normal's length remains 1 regardless of distance; this depth model and
  normalization cannot measure physical travel distance.
- Updated the stale MiDaS README to describe the current implementation.
- Five synthetic tests passed with NumPy/OpenCV. Live CUDA inference and FPS
  remain unverified: the project venv's referenced Python executable could
  not be launched in the review environment. No measured speedup is claimed.

## Teammate notebook corrections — September 12, 2026

- Revised `test_l1.ipynb` as a self-contained Colab geometry experiment.
  It now uses direct model inference with floating-point predictions, CUDA
  fp16 where available, and an adjustable input resolution (default 294).
- Replaced the 8-bit brightness-to-"meters" conversion with an explicitly
  approximate, positive inverse-depth-to-Z mapping in arbitrary units.
  Camera intrinsics remain assumed unless supplied; neither the mapping nor
  its normals constitute a calibrated reconstruction.
- Preserved the perspective Scharr/tangent estimator, added input validation,
  and replaced forced border normals with replicated-border derivatives.
  It returns raw XYZ normals, points, and a validity mask. The convention is
  X right, Y down, Z away; fronto-parallel normals point +Z. Negate them for
  camera-facing outward normals in a subsequent renderer.
- The live experiment shows synchronized RGB, Z proxy, and normal panels,
  a fixed center-pixel sample, and measured completed-loop rate after warmup.
  It preserves camera aspect ratio and stops camera tracks on exit/errors.
  An EMA stabilizes normalization bounds only, not moving-object geometry.
- Removed stale outputs containing unsupported metric measurements.
- Added `test_notebook_geometry.py`. Synthetic geometry checks passed on CPU
  and CUDA; simulated live-loop checks passed for panel synchronization,
  resolution changes, loop-rate reporting, and cleanup after inference errors.
  `python test_notebook_geometry.py --model-smoke` also passed using the cached
  model on CUDA, verifying float inference through normal extraction without
  downloading weights. The existing venv worked when run outside the sandbox.
- Actual Colab browser-camera behavior and live FPS remain unverified. These
  notebook checks do not establish a speedup for the separate `main.py` app.

## Level 02 notebook implementation — September 12, 2026

- User clarified that active development belongs in `test_l1.ipynb`, not
  `main.py`. Continue in the notebook. Preliminary desktop lighting edits
  (`lighting.py`, `main.py`, `test_lighting.py`, and a level-1 test adjustment)
  from before that clarification remain; the desktop lighting tests were not
  completed. Do not confuse their status with the validated notebook path.
- Added a self-contained PyTorch renderer using the notebook's perspective
  points and negated, camera-facing normals. It computes Lambertian diffuse,
  gated Blinn–Phong specular, softened inverse-square falloff, and ambient
  light in linear RGB before converting back to sRGB. Invalid geometry gets
  ambient only. Camera RGB is approximate albedo, so real lighting remains.
- Added live browser sliders for XYZ position, power, and specular strength,
  plus reset. Controls and JPEG frames travel together in one capture packet;
  the 2×2 display shows synchronized RGB, Z proxy, normals, and relighting.
  Coordinates are X right, Y down, Z away from the camera, in arbitrary units.
  Z slider range is -2 to 0.7, keeping the light before the nearest Z proxy.
- Added a fourth relit panel to the optional photo experiment. New lighting
  helper/test cells must run before photo/live cells. No repo imports needed.
- Notebook geometry and lighting checks passed on CPU and CUDA. The cached
  model smoke check passed on CUDA through inference, proxy depth, normals,
  and relighting. The simulated live loop passed frame/control alignment,
  resolution changes, completed-cycle reporting, and inference/render cleanup.
- `node test_notebook_controls.cjs` passed simulated browser checks for sliders,
  settings snapshots, reset, aspect ratio, stop, pending permission, and errors.
  Actual Google Colab browser/camera behavior and live throughput are still
  unverified. Next: run all notebook cells in Colab on GPU, try the light
  controls, and record completed-cycle rate at input sizes 294/392/518.

## Local notebook FPS optimization — September 12, 2026

- User moved away from hosted Colab and confirmed **local Jupyter / VS Code**.
  Active development remains in `test_l1.ipynb`; `main.py` was not changed in
  this optimization. The notebook now defaults to `LIVE_BACKEND="local"`.
- Added a local OpenCV window with XYZ/power/specular sliders, reset (R), quit
  (Q/Esc), and window-close cleanup. A capture thread retains only the newest
  frame and does not deliver the same frame ID twice. Each processed frame
  still receives its own inference and geometry; unread camera frames may be
  replaced to prevent accumulating latency. Local Level 3 hand tracking is
  not implemented; the existing optional Colab browser tracker is preserved.
- Added resolution presets: **fast = 196/320** (model target/capture width,
  default), **balanced = 224/480**, **detail = 294/640**. Camera aspect ratio is
  preserved; model dimensions follow the configured processor's patch rounding.
- Local output uses raw pixels in OpenCV, avoiding Colab round trips and JPEG
  encoding. Relighting-only is the default; aligned four-panel diagnostics are
  optional. Status updates are throttled, FPS appears in the local window, and
  CUDA event stage timings avoid per-stage synchronization. Timings overlap;
  download wait includes queued GPU work and must not be added to GPU stages.
- Added GPU-resident sampled-percentile normalization and a 5x5 bilateral
  filter, a tensor-return rendering/inference path, and OpenCV preprocessing
  with cached normalization tensors. `FAST_PREPROCESS=False` preserves the HF
  reference resize path; interpolation pixels may differ slightly in fast mode.
- **Measured tradeoff:** on the local MX550, a compute-only check at 480x360
  measured reference geometry at ~3.2 ms versus GPU smoothing/geometry at
  ~4.9 ms. `GEOMETRY_BACKEND="auto"` therefore benchmarks complete CPU and GPU
  geometry paths once during startup and selects the faster one; GPU processing
  is not assumed to be faster. Manual CPU/GPU overrides remain available.
- In the same local compute harness, depth inference/preprocessing/upsampling
  averaged ~58.4 ms (196), ~68.7 ms (224), and ~108.6 ms (294), after warmup.
  These are synthetic-image compute timings, not live-camera FPS or Colab FPS.
- Added `requirements-notebook.txt` for local kernel/plotting dependencies,
  `test_notebook_performance.py`, and `test_notebook_local.py`; updated the
  existing notebook and browser-control regression checks. Fixed a missing
  comma in the optional photo cell. Live camera and actual desktop window
  behavior still require the user's local run.
- Final checks passed: CPU/CUDA geometry and lighting; cached-model reference
  and fast paths at model targets 196/224/294; GPU/reference bilateral and
  normalization comparisons; automatic backend selection; fast/debug loop
  alignment; simulated local capture/window controls and cleanup; simulated
  browser controls; all Python notebook cells compile. Installed the missing
  local ipykernel/IPython/matplotlib dependencies into the project's `.venv`.

## Local Level 03 and hand intensity — September 12, 2026

- `test_l1.ipynb` now includes local MediaPipe VIDEO hand tracking, smooth XYZ
  mapping, palm-size Z calibration, loss/reacquisition handling and manual
  fallback. G toggles hand/manual, C recalibrates Z, R resets, Q/Esc stops.
- Opening the hand increases virtual light power toward 12; closing into a
  fist decreases it toward 0. Partial opening gives intermediate intensity.
  Four projected fingertip/wrist-to-PIP/wrist ratios estimate openness, with
  aspect correction, smoothstep mapping and a 180 ms power EMA. This heuristic
  is scale independent but palm rotation can affect it. Ambient light remains.
- In local gesture mode XYZ/Power sliders display tracked values; G enables
  manual editing. Specular is always adjustable. Invalid/clipped fingers hold
  intensity; lost palms hold the entire light. Colab retains manual power.
- Hand inference uses the same processed RGB frame as depth, adds a measured
  serial `hand` stage, and draws landmarks only on the output copy. The latest
  frame capture thread remains active; no hand-inference queue is introduced.
- Added a cached hand-model preparation cell and MediaPipe 0.10.32 to notebook
  requirements, using contrib OpenCV. The earlier dependency migration was
  blocked by Windows locking the loaded cv2 DLL; the user opted to test locally.
  No dependency installation was retried for the intensity change.
- `test_notebook_gestures.py` passes synthetic mapping, intensity endpoints and
  partial opening, smoothing, scale/aspect independence, loss/reacquisition,
  frame/control alignment, manual fallback and cleanup checks. Actual hand
  model inference, live gestures and FPS remain unverified by this assistant.

## Local Level 04 — dynamic shadows — September 12, 2026

- Added a self-contained `shadows` cell in `test_l1.ipynb`, used by the standard
  local camera loop and optional photo experiment. Existing notebook edits and
  the independent ONNX experiment are preserved. Colab and ONNX retain their
  existing rendering; `main.py` has not been ported or changed for this phase.
- `compute_shadow_visibility(depth, normals, valid, light, intrinsics, ...)`
  returns device-resident H×W float32 visibility. The geometry estimator now
  exposes its actual `(fx, fy, cx, cy)` intrinsics. Rays use normal-offset
  receivers, camera-frustum clipping and perspective-correct reciprocal-Z
  interpolation. Sampled ray intervals intersect finite depth slabs in batches
  of eight. Invalid/out-of-frame samples are non-blocking.
- `SHADOW_QUALITY="fast"` caps the receiver grid at 160 pixels wide and 32 ray
  samples; `"quality"` uses 320/64. Aspect ratio is preserved and both sample
  original-resolution depth. Defaults: bias 0.01×receiver Z, thickness
  0.03×occluder Z. Depth-aware 3×3 filtering and joint bilateral upsampling
  soften edges while limiting leakage across silhouettes. Every processed
  frame gets fresh shadows; no temporal mask reuse or inference queue is added.
- `relight_rgb(..., visibility=None)` remains compatible with existing callers.
  Visibility attenuates direct diffuse/specular before sRGB conversion, leaving
  ambient unchanged. Local/photo shadows default on; disabled shadows or zero
  power skip tracing. S toggles shadows, V toggles grayscale visibility; R
  retains its light/calibration behavior. The local status shows shadow state
  and a separate `shadow` stage. The standard output retains one final download.
- CPU/CUDA synthetic checks passed for flat/tilted planes, an analytically
  predicted foreground-blocker shadow (IoU 0.889), XYZ motion, invalid geometry,
  camera-plane clipping, changed resolutions, filtered edges and batch sizes.
  All-ones visibility preserves previous rendering exactly; all-zero visibility
  matches ambient-only rendering. Actual photo-cell integration passed.
- Cached CUDA model -> proxy geometry -> shadows -> relighting smoke passed.
  Existing geometry/lighting, fast/reference preprocessing, backend-selection,
  gesture/intensity, browser-control and simulated local-loop checks passed.
  Local tests cover same-frame geometry/light alignment, S/V, diagnostics,
  disabled/zero-power bypass and camera cleanup after a shadow failure.
- Saved `validation/phase4/benchmark.json` and a synthetic visual comparison.
  On the MX550, the fast shadow preset added mean compute costs of **8.29 ms**
  at 320×240, **9.05 ms** at 480×360 and **12.07 ms** at 640×480, from 15 warmup
  pairs plus 30 alternating off/on measured pairs. The 320×240 mean meets the
  10 ms target; on-path p95 was 16.20 ms. These are rendering-only timings,
  excluding depth inference, camera, hand tracking, download and display.
- Remaining acceptance: real camera/hand interaction, self-shadowing and
  stability on real estimated depth, and completed-loop FPS at each preset.
  No live FPS is claimed. Hidden/offscreen geometry cannot cast shadows,
  thin occluders may be missed, and filtered edges approximate softness rather
  than physical penumbrae. The README includes the manual validation sequence.
