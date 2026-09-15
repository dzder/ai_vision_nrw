# AI & Vision Challenge — NRW 8th Edition

Real-time monocular depth → 3D surface geometry → hand-controlled dynamic relighting, dynamic shadows, and multi-light volumetric rendering — from a single RGB webcam, no depth sensor.

Built for the **National Robotics Week (NRW), 8th Edition — AI & Vision Challenge**, organized by IEEE INSAT Student Branch and IEEE RAS INSAT Chapter.

**Team:** Ahmed Ben Khalfa · Mohamed Bachar Hmissi · Aziz Derbel · Melek Ben Dalloula

---

## Repository layout

```
phase00.ipynb   Environment & camera sanity checks
phase1.ipynb    Level 01 — Geometry Engine
phase2.ipynb    Level 02 — Dynamic Relighting
phase3.ipynb    Level 03 — Spatial Gesture Control
phase4.ipynb    Level 04 — Dynamic Shadows
phase5.ipynb    Level 05 — Multi-Light & Volumetric Scattering + bonuses
main.py         Desktop app — unified CLI entry point, all levels
capture.py      Webcam capture (numeric index or Windows device-location string)
hand_tracker.py MediaPipe hand tracking, 1–2 hands, palm → light mapping
lighting.py     CPU shading path
relight_gpu.py  GPU shading path, bloom
shadows.py      Screen-space ray-marched shadows, PCSS softening, multi-light
context.md      Running technical journal + literature citations
```

Each phase notebook is **self-contained** — it redefines the helpers it needs rather than importing from the previous phase. You do not have to run `phase1.ipynb` before `phase3.ipynb`. Going in order on a first pass is still the easiest way to understand how the pipeline builds up, level by level.

---

## Running the notebooks (Windows PowerShell)

### 1. Create the notebook environment

Install Python 3.11 (64-bit), then from the project folder:

```powershell
cd "path\to\project"
py -3.11 -m venv .venv-notebook
.\.venv-notebook\Scripts\python.exe -m pip install --upgrade pip
```

A separate environment avoids conflicts with the desktop app's own `.venv`.

Install one PyTorch build. For CPU only:

```powershell
.\.venv-notebook\Scripts\python.exe -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
```

For an NVIDIA GPU with a compatible driver, use this instead:

```powershell
.\.venv-notebook\Scripts\python.exe -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
```

Then install the remaining notebook dependencies and register the kernel:

```powershell
.\.venv-notebook\Scripts\python.exe -m pip install -r requirements-notebook.txt
.\.venv-notebook\Scripts\python.exe -m ipykernel install --user --name ai-vision-notebook --display-name "AI Vision (notebook)"
```

`requirements-notebook.txt` includes MediaPipe and `opencv-contrib-python`. Don't also install the desktop app's `requirements.txt` into this same environment — the two pull different OpenCV providers and installing both breaks the shared `cv2` module.

The first model run needs internet access to download `depth-anything/Depth-Anything-V2-Small-hf` from Hugging Face. Phase 3 onward also downloads `models/hand_landmarker.task` for hand tracking. Both are cached after the first run.

### 2. Open a notebook and select the kernel

**VS Code:** open the project folder, enable the Python and Jupyter extensions, open the phase notebook you want, and use *Select Kernel* → `AI Vision (notebook)` (or point directly at `.venv-notebook\Scripts\python.exe`).

**JupyterLab:**

```powershell
.\.venv-notebook\Scripts\python.exe -m pip install jupyterlab
.\.venv-notebook\Scripts\python.exe -m jupyter lab phase1.ipynb
```

Confirm the right environment is active from a scratch cell:

```python
import sys, torch
print(sys.executable)  # should end in .venv-notebook\Scripts\python.exe
print(torch.__version__, "CUDA available:", torch.cuda.is_available())
```

### 3. Running / testing a phase

**To validate a phase, run every cell in that notebook from top to bottom (`Run All`).** Each notebook is ordered so that running all cells in sequence exercises the full feature set for that level — geometry helpers, synthetic checks, the model load, and finally the live camera cell. There's no separate manual test procedure for the notebooks beyond this: if every cell completes without error and the live cell opens a working camera window, that phase is validated.

A few practical notes that apply across all phases:

- Skip any cell explicitly labeled *optional dependency setup* once you've installed `requirements-notebook.txt` — if you do run it anyway, restart the kernel before continuing.
- Live camera cells stay busy until you stop them (`Q`, `Esc`, or closing the window). Focus the OpenCV window for keyboard controls to register.
- The photo/still-image experiment cells (where present) work without a webcam — set the image path variable to an existing file and run.
- If a `NameError` appears for a helper that should already exist, you likely skipped a cell earlier in that notebook, or the kernel was restarted — re-run the notebook from the top.

---

## What each phase covers

| Phase | Level | What it adds |
|---|---|---|
| `phase00.ipynb` | — | Environment sanity checks: correct kernel, CUDA availability, camera access |
| `phase1.ipynb` | 01 — Geometry Engine | Depth inference, perspective unprojection, live surface normals synced to the camera feed |
| `phase2.ipynb` | 02 — Dynamic Relighting | Lambertian diffuse + Blinn-Phong specular shading with an interactive virtual point light |
| `phase3.ipynb` | 03 — Spatial Gesture Control | MediaPipe hand tracking maps palm position to the light's X/Y/Z |
| `phase4.ipynb` | 04 — Dynamic Shadows | Screen-space ray-marched, geometry-aware occlusion shadows |
| `phase5.ipynb` | 05 — Multi-Light & Volumetric Scattering | Two-hand tracking, two independent lights with independent shadows, volumetric haze, bloom, gesture-driven color, light tether |

---

## Live controls

| Key | Action |
|---|---|
| `G` | Toggle hand control vs. manual XYZ/Power sliders |
| `C` | Recalibrate hand depth at the current hand position |
| `R` | Reset the light and hand calibration |
| `S` | Toggle shadows |
| `X` | Toggle specular highlight |
| `V` | Toggle grayscale shadow-visibility view (white = illuminated) |
| `P` | Toggle PCSS-style shadow edge softening (phase4+) |
| `Q` / `Esc` | Stop processing and release the camera |

Hand control is on by default from phase3 onward: show one open palm facing the camera for a few frames, then move it to position the light. Opening your hand brightens it; closing it dims it. Manual sliders remain available at any time via `G`.

---

## Performance presets

| Preset | Model input size | Camera processing width |
|---|---|---|
| `fast` (default) | 196 | 320 |
| `balanced` | 224 | 480 |
| `detail` | 294 | 640 |

Re-run the setup and model cells after changing the preset. Lower resolution trades fine geometry detail for speed; aspect ratio is always preserved.

`GEOMETRY_BACKEND="auto"` benchmarks CPU vs. GPU smoothing/normal-extraction at startup and picks the faster path automatically. Stage timings and completed-loop FPS are reported after warmup and reflect application throughput, not raw sensor-to-screen latency.

---

## Shadow implementation

Each processed frame traces rays from the visible depth surface toward the current light position, clips to the camera view, and interpolates reciprocal depth along the projected ray. Occlusion is tested in bounded batches; depth-aware filtering and upsampling soften edges while limiting leakage across foreground silhouettes. Visibility scales direct diffuse and specular light only — ambient is untouched. No shadow history carries between frames.

| Setting | Fast | Quality |
|---|---|---|
| Receiver grid width | 160 px | 320 px |
| Ray samples | 32 | 64 |

`SHADOW_BIAS` offsets receivers along their outward normal to avoid self-shadowing; `SHADOW_THICKNESS` gives occluders finite thickness — both expressed as fractions of camera Z.

Measured shadow overhead (synthetic geometry, fast preset, 15 warmup + 30 measured pairs):

| Processing size | Shadows off | Shadows on | Added mean time |
|---|---|---|---|
| 320×240 | 1.55 ms | 9.85 ms | 8.29 ms |
| 480×360 | 3.16 ms | 12.21 ms | 9.05 ms |
| 640×480 | 5.36 ms | 17.43 ms | 12.07 ms |

These are rendering-compute costs only — they exclude depth inference, camera capture, hand tracking, and display. Reproduce with:

```powershell
.\.venv-notebook\Scripts\python.exe test_notebook_shadows.py --benchmark --output-dir validation/phase4
```

---

## Desktop app (`main.py`)

The desktop app is the unified, submittable entry point covering all levels through a single CLI.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
python main.py --level 5 --device cuda --camera 0
```

Use `--index-url https://download.pytorch.org/whl/cpu` instead for a CPU-only setup.

| Flag | Purpose |
|---|---|
| `--level {1..5}` | Which feature set to run (default 4) |
| `--device {cpu,cuda}` | Inference device |
| `--camera` | Numeric index, or a Windows device-location string for a specific physical camera |
| `--input-size` | Depth model input resolution — must be a multiple of 14, minimum 140 |
| `--shadows`, `--shadow-quality` | Toggle and tune shadow ray marching |
| `--bloom`, `--haze`, `--tether` | Toggle Level ∞ bonus effects |
| `--diag` | Multi-panel debug view instead of the lean single-panel display |
| `--benchmark-frames N` | Run N frames headless and report FPS / stage timing |

CUDA runs in fp16; CPU runs in fp32. First run downloads the Hugging Face checkpoint; subsequent runs reuse the cache.

---

## Validation (no camera required)

```powershell
.\.venv-notebook\Scripts\python.exe test_notebook_geometry.py
.\.venv-notebook\Scripts\python.exe test_notebook_local.py
.\.venv-notebook\Scripts\python.exe test_notebook_gestures.py
.\.venv-notebook\Scripts\python.exe test_notebook_shadows.py
.\.venv-notebook\Scripts\python.exe test_notebook_lighting_cull.py
.\.venv-notebook\Scripts\python.exe test_notebook_performance.py --benchmark
```

Add `--model-smoke` to the geometry or shadows scripts for an optional check using cached model weights. Browser-control checks use `node test_notebook_controls.cjs` (Node.js only needed for that check). On the desktop side, `test_main_pipeline.py` and `test_lighting.py` validate the CLI pipeline against a fake camera/depth/hand input — no webcam required.

---

## Architecture notes

- Depth is affine-ambiguous relative inverse depth, not metric. `D = 1 - clip((inverse - p2) / (p98 - p2), 0, 1)` reverses ordering into a usable relief map — it is not a physical reciprocal distance.
- Surface normals come from a Gaussian-smoothed depth map through Scharr/Sobel gradients, unprojected via camera intrinsics, then unit-normalized.
- The hand-tracked light position is converted once into a single shared coordinate convention and reused identically by both the shading stage and the shadow ray-marcher, so the rendered light and its cast shadow always agree spatially.
- Multi-light shadows (phase5 / `--level 5`) share one depth buffer across both lights rather than computing a separate shadow map per light.
- Depth inference runs on its own thread, overlapping with rendering rather than blocking it.

---

## Technical references

- **McGuire, M. & Mara, M. (2014)** — *Efficient GPU Screen-Space Ray Tracing* — basis for the shadow ray-marching.
- **Cowan & Khattak** — *Screen Space Point Sampled Shadows* — multi-light, shared-depth-buffer shadow architecture.
- **Fernando, R. (2005)** — *Percentage-Closer Soft Shadows* — distance-dependent shadow-edge softening.
- **Kawase, M. (2003)**, GDC — *Frame Buffer Postprocessing Effects in DOUBLE-S.T.E.A.L* — base multi-pass bloom technique.
- **Bjørge, M. (2015)**, SIGGRAPH — *Bandwidth-Efficient Rendering* — dual-filter downsample/upsample refinement used in the bloom implementation.

See `context.md` for the full running journal of design decisions, benchmarks, and fixes.

---

## Troubleshooting

**`py` is missing / Python 3.11 not found** — install Python 3.11 via the Windows Python launcher, reopen PowerShell, repeat setup.

**`Unable to create process from .venv`** — the environment is tied to a removed Python install; use `.venv-notebook` and select its kernel instead.

**`ModuleNotFoundError`** — check `sys.executable` in the notebook, confirm the right kernel is selected, install `requirements-notebook.txt` with that interpreter, restart the kernel.

**OpenCV window errors / conflicting `cv2` packages** — stop the kernel and repair the environment:

```powershell
.\.venv-notebook\Scripts\python.exe -m pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python-headless
.\.venv-notebook\Scripts\python.exe -m pip install --force-reinstall "opencv-contrib-python>=4.8,<4.12" "numpy>=1.24,<2"
```

**Camera cannot open** — close other camera apps, enable camera access for desktop apps in Windows Settings, try a different camera index or a Windows device-location string.

**CUDA unavailable or processing slow** — verify the installed PyTorch build matches your hardware. CPU mode works; start with the `fast` preset and disable hand control/shadows if needed.

**Model download fails** — restore internet access and re-run the model cell; manual sliders still work without it, but depth inference itself requires the depth model.

---

## Known limitations

- Depth is relative, not metric — absolute distances are approximate without calibrated intrinsics and a metric depth source.
- Hand detection is less reliable when strongly backlit (bright light source directly behind the subject).
- Hidden or off-camera geometry cannot cast shadows, since shadows are computed only from the visible depth layer.
- CPU-only inference is significantly slower than CUDA; an OpenVINO backend was explored for CPU acceleration.

---

## Submission deliverables

Per the challenge specification, this repository is one of three required deliverables:
1. **Source code** (this repo) — the desktop app launches directly via `python main.py`.
2. **Technical Architecture Brief** — model choice, normal-extraction formulas, temporal filtering strategy, optimization notes.
3. **Pitch Deck** — pipeline overview, trade-offs, FPS benchmarks, live demo configuration.

## Acknowledgments

Organized by the IEEE INSAT Student Branch and IEEE RAS INSAT Chapter as part of National Robotics Week, 8th Edition.