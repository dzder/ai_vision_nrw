# AI & Vision: Geometry and Dynamic Relighting

## Run the notebook (Windows PowerShell)

Open **[test_l1.ipynb](test_l1.ipynb)** for Levels 1–4: geometry, relighting,
hand control, and dynamic shadows. Use a local desktop session for the webcam
and OpenCV window. The photo experiment works without a webcam.

### 1. Create a notebook environment

Install Python 3.11 (64-bit), then run these commands from the project folder.
Replace the path if your checkout is elsewhere:

```powershell
cd "D:\AI VISION"
py -3.11 -m venv .venv-notebook
.\.venv-notebook\Scripts\python.exe -m pip install --upgrade pip
```

This uses a separate environment because the existing `.venv` may point to a
Python installation that is no longer available. Activation is not required.

Install **one** PyTorch build. For CPU (no NVIDIA GPU required):

```powershell
.\.venv-notebook\Scripts\python.exe -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
```

For an NVIDIA GPU with a compatible driver, use this command instead of the CPU
command:

```powershell
.\.venv-notebook\Scripts\python.exe -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
```

Then install the notebook dependencies and register its kernel:

```powershell
.\.venv-notebook\Scripts\python.exe -m pip install -r requirements-notebook.txt
.\.venv-notebook\Scripts\python.exe -m ipykernel install --user --name ai-vision-notebook --display-name "AI Vision (notebook)"
```

Use `requirements-notebook.txt` for this environment. It includes MediaPipe and
`opencv-contrib-python`; the desktop app's `requirements.txt` installs a
different OpenCV provider. Installing both can break the shared `cv2` module.
The first model run needs internet access to download
`depth-anything/Depth-Anything-V2-Small-hf` from Hugging Face. Hand control also
downloads `models/hand_landmarker.task`; subsequent runs reuse cached models.

### 2. Open the notebook and select the kernel

**VS Code:** open the project folder, install/enable the Python and Jupyter
extensions, open `test_l1.ipynb`, and use **Select Kernel** to choose
**AI Vision (notebook)** or `.venv-notebook\Scripts\python.exe`.

**JupyterLab:** install and launch it from the project folder:

```powershell
.\.venv-notebook\Scripts\python.exe -m pip install jupyterlab
.\.venv-notebook\Scripts\python.exe -m jupyter lab test_l1.ipynb
```

Select **AI Vision (notebook)** as the kernel. In either editor, you can check
the selected environment in a temporary notebook cell:

```python
import sys, torch
print(sys.executable)  # Should end in .venv-notebook\Scripts\python.exe
print(torch.__version__, "CUDA available:", torch.cuda.is_available())
```

### 3. Run the cells in order

Run individual cells with **Shift+Enter**. Skip the first code cell, labeled
**Optional dependency setup**, when you installed the requirements above. If
you do run that installer cell, restart the kernel before running other cells.

1. Run the imports/settings cell (`import io`). Leave `LIVE_BACKEND = "local"`
   and `PERFORMANCE_PRESET = "fast"` for the first run. The device is selected
   automatically: CUDA when available, otherwise CPU.
2. Run the geometry helpers and synthetic checks, then the Level 2 lighting
   helper and lighting checks.
3. Run the model cell that prints **Loading Depth Anything V2 Small...**.
   Wait for the download and initialization to finish.
4. Run the Level 4 shadow helper cell (`# Self-contained Level 4`). Both photo
   and live rendering depend on it.
5. Optionally run the photo cell. Set `PHOTO_PATH` to an existing image, for
   example `PHOTO_PATH = r"D:\Pictures\scene.jpg"`. Skip it for webcam use.
6. Run the hand-model preparation cell (`# Run once before the live cell`).
   Set `HAND_CONTROL = False` in setup if you only want manual sliders.
7. Run the large live cell that starts with `def start_colab_webcam():` and ends
   with `run_live()`. Despite its first function name, it uses the selected
   local backend and opens the webcam in an OpenCV window.

The live cell stays busy until you stop it. Focus the OpenCV window to use its
keyboard controls. Skip the final **Optional ONNX experiment** for normal use;
it has separate dependencies in `requirements-onnx-experiment.txt` and requires
CUDA. It is not needed to run the notebook.

### Live controls

| Control | Action |
|---|---|
| **Q**, **Esc**, or close window | Stop processing and release the camera |
| **G** | Switch between hand control and manual XYZ/Power sliders |
| **C** | Recalibrate hand depth at the current hand position |
| **R** | Reset the light and hand calibration |
| **S** | Toggle shadows |
| **V** | Toggle grayscale shadow visibility (white means illuminated) |
| Specular slider | Adjust highlights in either control mode |

Hand control is enabled by default. Show one open palm facing the camera for
three processed frames, then move it to position the light. Open your hand to
brighten the light; close it to dim. Press **G** to use sliders manually.
If the hand model or tracking fails, manual controls remain available.

### Performance and display

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
clip. Local/photo shadows use the visible depth layer, so hidden or off-camera
objects cannot cast shadows. `SHADOW_QUALITY="fast"` is the default; `"quality"`
adds denser tracing at higher cost. Stop the live cell before changing setup,
then rerun setup and any affected cells before restarting live processing.

The optional `LIVE_BACKEND="colab"` path uses browser camera access and hand
tracking, JPEG output, and throttled text updates. It retains Level 3 rendering;
Level 4 live shadows and the keyboard controls above apply to local mode.
Actual live-camera FPS must be measured on your machine.

### Shadow implementation and measured overhead

`SHADOW_QUALITY="fast"` caps the receiver grid at 160 pixels wide and 32 ray
samples; `"quality"` uses 320/64. This setting is independent of the camera/model
preset. Both sample original-resolution depth and preserve aspect ratio.
`SHADOW_BIAS=0.01` offsets receivers along outward normals;
`SHADOW_THICKNESS=0.03` gives occluders finite thickness, as fractions of camera Z.

Each processed frame traces toward its current light, clips rays to the camera
view, and interpolates reciprocal Z along projected segments. Bounded batches
test depth intervals for occlusion. Depth-aware filtering and upsampling soften
edges while limiting leakage across foreground silhouettes. Visibility scales
direct diffuse/specular lighting; ambient stays unchanged. No shadow history is
reused. Disabled shadows or zero power skip tracing. Thin occluders may be missed;
smoothed boundaries are not physical area-light penumbrae.

**V** temporarily replaces relighting or four-panel diagnostics with visibility;
V again restores the chosen layout. **R** preserves shadow toggles. For photos,
set `SHOW_SHADOW_VISIBILITY=True` to inspect visibility in the fourth panel.
The independent ONNX comparison retains its existing rendering; S/V apply to
the standard local loop. The `shadow` stage includes tracing and filtering;
CUDA stage timings overlap and must not be added together.

The saved [MX550 benchmark](validation/phase4/benchmark.json) used the fast shadow
preset on synthetic geometry, 15 warmup pairs and 30 measured pairs, alternating
shadow-off/on order:

| Processing size | Shadows off | Shadows on | Added mean time |
|---|---:|---:|---:|
| 320×240 | 1.55 ms | 9.85 ms | 8.29 ms |
| 480×360 | 3.16 ms | 12.21 ms | 9.05 ms |
| 640×480 | 5.36 ms | 17.43 ms | 12.07 ms |

These are rendering compute costs, excluding depth inference, camera, hand
tracking, image download and display. The 320×240 mean meets the 10 ms overhead
target; individual cycles vary (on-path p95: 16.20 ms). They are not live FPS.
See the [synthetic shadow comparison](validation/phase4/shadow_comparison.png).
Reproduce the checks and save a new report with:

```powershell
.\.venv-notebook\Scripts\python.exe test_notebook_shadows.py --benchmark --output-dir validation/phase4
```

For live acceptance, start with fast and a foreground subject against a visible
wall. Compare S off/on; move the light laterally and closer/farther with your hand,
then inspect V. Check that lateral shadows move opposite the light, foreground
surfaces avoid broad self-shadowing, and moving subjects leave no persistent
trails. Check open/closed-hand intensity, G/C/R, and camera quit/restart. Record
completed-loop FPS and the shadow stage with hand tracking enabled at each preset.
This hands-on validation is pending.

### Troubleshooting

- **`py` is missing / Python 3.11 is not found:** install Python 3.11 with the
  Windows Python launcher, reopen PowerShell, and repeat setup.
- **`Unable to create process` from `.venv`:** use the new `.venv-notebook`
  environment and select its kernel. An environment tied to a removed Python
  installation cannot run the notebook.
- **`ModuleNotFoundError`:** check `sys.executable` in the notebook, select the
  notebook kernel, and install `requirements-notebook.txt` with that interpreter.
  Restart the kernel after package changes.
- **OpenCV window errors or conflicting `cv2` packages:** stop the kernel,
  repair the notebook environment below, and restart it:

  ```powershell
  .\.venv-notebook\Scripts\python.exe -m pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python-headless
  .\.venv-notebook\Scripts\python.exe -m pip install --force-reinstall "opencv-contrib-python>=4.8,<4.12" "numpy>=1.24,<2"
  ```

- **Camera cannot open:** close other camera apps, enable camera access for
  desktop apps in Windows settings, and try `CAMERA_INDEX = 1` in setup. Stop
  the live cell before rerunning setup and starting it again.
- **`NameError` for a helper or model:** rerun the required cells in the order
  above, including the Level 4 shadow helper after a kernel restart.
- **CUDA unavailable or processing slow:** check the kernel and installed
  PyTorch build. CPU mode works; start with `fast`, and disable hand control
  or shadows if needed. A requested camera FPS is not the processing FPS.
- **Model download fails:** restore internet access and rerun the model cell
  or hand-model preparation cell. Manual sliders work without the hand model;
  depth inference still requires the depth model.

### Notebook validation (no camera required)

Run from the project folder after installing the notebook requirements:

```powershell
.\.venv-notebook\Scripts\python.exe test_notebook_geometry.py
.\.venv-notebook\Scripts\python.exe test_notebook_local.py
.\.venv-notebook\Scripts\python.exe test_notebook_gestures.py
.\.venv-notebook\Scripts\python.exe test_notebook_shadows.py
```

For optional checks using cached model weights, add `--model-smoke` to
`test_notebook_geometry.py` or `test_notebook_shadows.py`. Run
`test_notebook_performance.py --benchmark` with the same interpreter for local
compute timings. Browser-control checks use `node test_notebook_controls.cjs`;
Node.js is only needed for that development test.

## Local geometry application

The local app also has preliminary level 2 edits. Use `python main.py --level 1`
for the original two-panel geometry/probe workflow described below. The notebook
validation above does not validate the desktop relighting implementation.

OpenCV webcam → Depth Anything V2 Small → Sobel unit normals → synchronized
camera/normal display. Project requirements are in [context.md](context.md).

## Desktop app setup (Windows PowerShell)

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
