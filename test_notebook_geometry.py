"""Check the notebook's actual helper cells without Colab or model downloads."""
import ast
import argparse
import json
import time
from collections import deque
from functools import partial
from pathlib import Path
from types import SimpleNamespace


def check_live_loop(notebook, geometry):
    """Exercise the actual live loop with deterministic frames and display handles."""
    from PIL import Image

    class DisplayHandle:
        def __init__(self):
            self.updates = []

        def update(self, value):
            self.updates.append(value)

    np = geometry["np"]
    namespace = dict(geometry)
    namespace.update(Image=Image, DisplayImage=lambda **kw: SimpleNamespace(**kw), time=time, deque=deque,
                     DEVICE="cpu", ASSUMED_HFOV_DEG=60, JPEG_QUALITY=75,
                     SHOW_DIAGNOSTICS=True, STATUS_INTERVAL=0, PROFILE_STAGES=True,
                     LIVE_BACKEND="colab")
    source = next("".join(cell["source"]) for cell in notebook["cells"] if "live" in cell["metadata"].get("tags", []))
    tree = ast.parse(source)
    # Load functions without invoking the interactive entry point.
    tree.body = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))]
    exec(compile(tree, "notebook live functions", "exec"), namespace)
    handles, closed = [], []

    def display(value, display_id):
        handle = DisplayHandle()
        handles.append(handle)
        return handle

    frames = [np.full((24, 32, 3), 25 + i, np.uint8) for i in range(4)]
    frames.append(np.full((30, 40, 3), 29, np.uint8))
    lights = [dict(x=-0.6+i*0.2, y=-0.4, z=0.1*i, power=float(i), specular=0.3)
              for i in range(len(frames))]
    pending = iter(list(zip(frames, lights)) + [None])
    rendered = []

    def relight(frame, normals, points, valid, light, **kwargs):
        result = geometry["relight_rgb"](frame, normals, points, valid, light, **kwargs)
        rendered.append((frame.copy(), dict(light), result.cpu().numpy().copy()))
        return result

    namespace.update(
        display=display, Markdown=lambda value: value,
        start_webcam=lambda: True, stop_webcam=lambda: closed.append(True),
        get_frame=lambda: (lambda p: None if p is None else (*p, dict(mode="manual", status="Manual", age_ms=0)))(next(pending)),
        infer_relative_inverse=lambda frame, **kw: geometry["torch"].full(frame.shape[:2], 0.5),
        relight_rgb=relight,
        encode_live_image=lambda picture: picture.copy(),
    )
    namespace["run_live"]()
    assert len(closed) == 1
    assert len(handles[1].updates) == len(frames)
    for frame, light, composite, (render_frame, render_light, relit) in zip(frames, lights, handles[1].updates, rendered):
        actual = np.asarray(composite)
        h, w = frame.shape[:2]
        assert actual.shape == (2 * (h + 32), 2 * w, 3)
        np.testing.assert_array_equal(actual[32:h+32, :w], frame)
        np.testing.assert_array_equal(render_frame, frame)
        assert render_light == light
        np.testing.assert_array_equal(actual[h+64:, w:], relit)
        # Float cancellation can land on either side of the 127.5 RGB midpoint.
        np.testing.assert_allclose(actual[h+64:, :w], np.broadcast_to([128, 128, 255], (h, w, 3)), atol=1, rtol=0)
    assert "cycles/s" in handles[0].updates[-1]
    assert "arbitrary units" in handles[0].updates[-2]
    assert "Light XYZ" in handles[0].updates[-2]

    # A model failure must still stop camera tracks and propagate the error.
    def fail_inference(frame, **kw):
        raise RuntimeError("synthetic inference failure")

    namespace.update(get_frame=lambda: (frames[0], lights[0], dict(mode="manual", status="Manual", age_ms=0)),
                     infer_relative_inverse=fail_inference)
    try:
        namespace["run_live"]()
    except RuntimeError as error:
        assert str(error) == "synthetic inference failure"
    else:
        raise AssertionError("The live loop swallowed an inference failure.")
    assert len(closed) == 2
    # Rendering failures follow the same cleanup path as model failures.
    def fail_render(*args, **kw):
        raise RuntimeError("synthetic rendering failure")

    namespace.update(infer_relative_inverse=lambda frame, **kw: geometry["torch"].full(frame.shape[:2], 0.5),
                     relight_rgb=fail_render)
    try:
        namespace["run_live"]()
    except RuntimeError as error:
        assert str(error) == "synthetic rendering failure"
    else:
        raise AssertionError("The live loop swallowed a rendering failure.")
    assert len(closed) == 3
    # Fast display keeps the same frame/light association through a shape change.
    pending = iter(list(zip(frames, lights)) + [None])
    namespace.update(SHOW_DIAGNOSTICS=False, STATUS_INTERVAL=1000, relight_rgb=relight,
        get_frame=lambda: (lambda p: None if p is None else (*p, dict(mode="manual", status="Manual", age_ms=0)))(next(pending)))
    namespace["run_live"]()
    assert len(closed) == 4
    assert len(handles[-2].updates) == 2  # First status, then cleanup; no per-frame text.
    assert len(handles[-1].updates) == len(frames)
    for actual, (frame, light, relit) in zip(handles[-1].updates, rendered[-len(frames):]):
        assert actual.shape == frame.shape
        np.testing.assert_array_equal(actual, relit)
    print("Live-loop checks passed: fast/debug frame alignment, light updates, resize, rate/status throttling, and error cleanup.")


def check_cached_model(notebook, geometry):
    """Run the notebook's actual inference cell using cached weights only."""
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    torch, np = geometry["torch"], geometry["np"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    namespace = dict(geometry)
    namespace.update(
        DEVICE=device, DTYPE=torch.float16 if device.type == "cuda" else torch.float32,
        INPUT_SIZE=294, MODEL_ID="depth-anything/Depth-Anything-V2-Small-hf",
        AutoImageProcessor=SimpleNamespace(from_pretrained=partial(AutoImageProcessor.from_pretrained, local_files_only=True)),
        AutoModelForDepthEstimation=SimpleNamespace(from_pretrained=partial(AutoModelForDepthEstimation.from_pretrained, local_files_only=True)),
    )
    source = next("".join(cell["source"]) for cell in notebook["cells"] if "model" in cell["metadata"].get("tags", []))
    exec(compile(source, "notebook model cell", "exec"), namespace)
    frame = np.random.default_rng(4).integers(0, 256, (120, 160, 3), dtype=np.uint8)
    raw = namespace["infer_relative_inverse"](frame)
    assert raw.shape == frame.shape[:2] and raw.dtype == np.float32 and np.isfinite(raw).all()
    proxy = namespace["RelativeGeometryDepth"]()(raw)
    estimator = namespace["ScharrNormalEstimator"](*raw.shape, device=device)
    normals, points, valid = estimator.compute_normals_and_coords(proxy)
    assert torch.isfinite(normals).all() and torch.isfinite(points).all() and valid.any()
    relit = namespace["relight_rgb"](frame, normals, points, valid)
    assert relit.shape == frame.shape and relit.dtype == np.uint8
    print(f"Cached-model smoke check passed on {device}: float inference -> Z proxy -> normals -> relighting. No camera/FPS benchmark.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-smoke", action="store_true", help="Also check inference using cached model files; never download weights.")
    args = parser.parse_args()
    notebook = json.loads(Path(__file__).with_name("test_l1.ipynb").read_text(encoding="utf-8"))
    namespace = {}
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        # IPython installation magic is intentionally not plain Python.
        source = "\n".join("pass" if line.startswith("%pip ") else line for line in source.splitlines())
        code = compile(source, f"test_l1.ipynb cell {index}", "exec")
        if set(cell["metadata"].get("tags", [])) & {"geometry", "synthetic-tests", "lighting", "lighting-tests"}:
            exec(code, namespace)
    torch = namespace["torch"]
    if torch.cuda.is_available():
        namespace["run_synthetic_checks"]("cuda")
        namespace["run_lighting_checks"]("cuda")
    check_live_loop(notebook, namespace)
    if args.model_smoke:
        check_cached_model(notebook, namespace)
    print("All notebook Python cells compile.")


if __name__ == "__main__":
    main()
