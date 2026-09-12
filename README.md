# AI & Vision: Geometry and Dynamic Relighting

## Level 2 in a local Jupyter / VS Code notebook

The active notebook is **`test_l1.ipynb`** (the filename is retained). It is
self-contained. Install `requirements-notebook.txt`, open the notebook in
VS Code/Jupyter, and select the project's `.venv` Python kernel. Run the cells
in order; the photo experiment is optional. `LIVE_BACKEND="local"` is the default.
The live cell opens an OpenCV window with XYZ, power, and specular sliders.
Press **Q/Esc** or close the window to stop; **R** resets the light.

Choose `PERFORMANCE_PRESET` in the setup cell:

| Preset | Model target | Camera processing width |
|---|---:|---:|
| `fast` (default) | 196 | 320 |
| `balanced` | 224 | 480 |
| `detail` | 294 | 640 |

Re-run setup and model cells after changing the preset. Lower resolution trades
fine geometry for speed. Aspect ratio is preserved. Local display sends raw
pixels directly to OpenCV; a capture thread retains only the latest frame.
Every processed frame gets fresh depth, normals, and shading. The default
output is relighting only; `SHOW_DIAGNOSTICS=True` restores four aligned panels.
`GEOMETRY_BACKEND="auto"` compares CPU and GPU smoothing plus normal extraction
at startup and uses the faster path. Stage timings and completed-loop FPS are
reported after warmup; they do not measure sensor-to-screen latency.

The new lighting cell uses the notebook's perspective XYZ points and normals
for Lambertian diffuse and Blinn–Phong specular shading, distance falloff, and
linear-RGB light calculations. X increases rightward, Y downward, and Z toward
the scene. Sliders take effect on the next processed frame.

Light positions and depth use arbitrary units. Existing real lighting remains
in the camera image; this adds virtual illumination. Reduce power if highlights
clip. Cast shadows are not implemented. Local mode currently uses manual
controls; the optional `LIVE_BACKEND="colab"` path retains browser hand tracking
for Level 3, loaded only when enabled. Colab uses JPEG output and throttled text
updates. Actual live-camera FPS has not yet been measured for the updated path.

Notebook validation (no camera required):

```powershell
python test_notebook_geometry.py
python test_notebook_geometry.py --model-smoke  # Cached weights only
python test_notebook_performance.py --model-smoke
python test_notebook_performance.py --benchmark  # Local compute only, no camera
python test_notebook_local.py
node test_notebook_controls.cjs
```

CPU/CUDA geometry and shading checks, a cached CUDA model-to-render smoke
check, fast/reference preprocessing and smoothing comparisons, simulated local
capture/display checks, and browser-control checks are available. The browser
test requires Node.js only for development; the live notebook does not need it.

## Local geometry application

The local app also has preliminary level 2 edits. Use `python main.py --level 1`
for the original two-panel geometry/probe workflow described below. The notebook
validation above does not validate the desktop relighting implementation.

OpenCV webcam → Depth Anything V2 Small → Sobel unit normals → synchronized
camera/normal display. Project requirements are in [context.md](context.md).

## Setup (Windows PowerShell)

Use Python 3.10 or 3.11 with the matching PyTorch build for your machine:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
python main.py
```

For CPU, use `https://download.pytorch.org/whl/cpu` in the PyTorch install
command. First run downloads the Hugging Face checkpoint
`depth-anything/Depth-Anything-V2-Small-hf`; subsequent runs reuse the cache.

## Speed and benchmarking

The default model resize target remains 518 for the existing quality baseline.
Try 294 for reduced inference work, or 392 as an intermediate setting:

```powershell
python main.py --device cuda --input-size 518 --benchmark-frames 150
python main.py --device cuda --input-size 294 --benchmark-frames 150
python main.py --device cuda --input-size 392 --benchmark-frames 150
```

Use the same camera, scene, capture dimensions, power settings, and point
selection state for each run. Compare surface detail and tracking stability as
well as FPS. Smaller inputs can lose fine geometry or increase normal artifacts.
The resize target must be a multiple of 14 and at least 140. The processor
preserves aspect ratio, so actual model H×W can differ from the target; startup
and the footer report it. Reducing camera dimensions alone does not set model
resolution. Other options: `--camera 1`, `--width 320 --height 240`,
`--device cpu`, and `--strength 60`.

CUDA uses fp16; CPU uses fp32. Depth processing includes preprocessing, device
transfer, inference, interpolation and normalization. Returning a CPU depth map
completes the relevant GPU work before the depth timer stops. Three warm-up
inferences are excluded. The footer shows rolling throughput and stage times;
exit prints mean times and overall FPS for all completed iterations. The
normals/probe stage includes display assembly; display time includes text,
OpenCV submission and GUI event handling. These are application throughput
measurements, not sensor-to-screen latency. Each displayed camera image
accompanies its own predicted normals. Camera buffering remains backend
dependent despite requesting a one-frame buffer.

## Inspect a point

- Left-click a textured feature on either image to select it. The point follows
  motion using pyramidal Lucas–Kanade flow, with forward/backward and photometric
  checks. Move slowly, especially at low FPS. This tracks a local image feature,
  not a recognized object; occlusion, low texture, or large motion can lose it or
  cause drift. Re-select if it drifts; detected failures show “Tracking lost”.
- Both panels show the selected point and its normal's XY projection, drawn
  with a fixed 60-pixel vector scale. An arrow pointing mostly along Z has a short
  XY projection even though its 3D length is still 1.
- The footer reports the normalized mean direction in a 5×5 patch, its length,
  angle from the selection reference, and median relative depth in that patch.
- **B** resets the direction reference; **C** or right-click clears the point;
  **Q**, Escape, or closing the window exits.

**A normal describes surface orientation, not distance. Its normalized length
is always 1.** For an ideal surface with unchanged orientation, increasing its
distance should not increase its normal's length. The displayed direction can
still change because this implementation uses image-space relative geometry,
and because of perspective, prediction noise, or changing scene normalization.
The angle readout is therefore an exploratory proxy, not a calibrated angle
measurement. Select a textured patch away from object boundaries for inspection.

The relative-depth value is normalized independently for every frame. It ranks
near/far within that frame, but cannot reliably quantify how far an object moved
between frames. Metric distance and physical surface normals require calibrated
camera intrinsics plus suitable metric depth (for example, a depth camera or
a validated metric reconstruction pipeline).

## Architecture and geometry

- `capture.py`: unmodified H×W×3 uint8 BGR camera frames.
- `depth.py`: model inference → aligned H×W float32 relative-depth proxy.
- `normals.py`: gradients, unit XYZ normals, RGB conversion, patch sampling.
- `probe.py`: clicked-point tracking and normal arrow drawing.
- `main.py`: serial processing, CLI, display and measured stage timings.

The model estimates affine-ambiguous inverse depth. We compute
`D = 1 - clip((inverse - p2) / (p98 - p2), 0, 1)`. This reverses ordering;
it is not a metric reciprocal. Flat predictions produce a flat normal map.
A 5×5 Gaussian blur suppresses spatial noise, followed by 3×3 Sobel derivatives
with scale 1/8. We normalize `(-strength*Dx, -strength*Dy, 1)` to unit length.
Coordinates are X right, Y down, Z positive. Strength (default 40) controls the
unknown depth-to-pixel scale. Normals are image-space height-field estimates,
not calibrated perspective normals. Colors are `round(255*(N+1)/2)` in RGB,
converted to BGR at display. Flat normals are RGB (128, 128, 255).

There is no temporal depth filtering yet; per-frame normalization may flicker.
Adding filtering needs motion compensation to avoid lag and ghosting around
moving objects. The point tracker does not temporally smooth the depth map.

Official references:
- [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2)
- [Hugging Face Depth Anything](https://huggingface.co/docs/transformers/model_doc/depth_anything)

## Validation

Run synthetic geometry, point-tracking, and simulated display-loop checks:

```powershell
python -m unittest test_geometry -v
```

Five synthetic checks passed during this review using an alternative bundled
Python runtime with NumPy/OpenCV. They cover flat/sloped normals, unit length,
depth-offset invariance, border sampling, point movement/loss, mouse mapping,
and a simulated display loop with benchmark exit and camera cleanup.

The earlier 4.6 FPS result in `context.md` predates the direct-model optimization.
No new camera/CUDA FPS result has been measured in this review. The existing
`.venv` points to an unavailable Python 3.10 executable in the review environment.
