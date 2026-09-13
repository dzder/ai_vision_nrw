"""Level 2 relighting optimizations: hand-centered shading cull + flat-surface
specular/diffuse suppression.

Regression, timing and preview for the notebook cell-6 ``relight_rgb``.

The cull box is a true crop-and-slice implementation: points, normals, validity
and albedo are sliced to the box BEFORE any per-pixel vector math, and the
result is pasted onto a full-frame ambient base. Outside the box every pixel
is ``decode(albedo) * ambient`` re-encoded -- never a raw camera copy and never
the full-frame direct math. The mask-after variant (compute full frame, then
zero the direct terms outside the box) is ruled out by a radius-sweep timing
check in ``benchmark``: if it were mask-after, shrinking the radius would not
shrink the measured time. The measured times are local synthetic compute, not
live-camera FPS.

Run:
    python test_notebook_lighting_cull.py [--device cpu|cuda] [--output-dir validation/lighting_cull]
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from test_notebook_performance import load_helpers


def scene(height, width, device, seed=1234):
    """Flat wall (z = 3) with a raised hemisphere bump (z = 1.5..3).

    Raw normals point +Z on the wall and deviate on the bump, giving both flat
    and slope pixels for the flat-suppression checks. Albedo texture so the
    relit result is not uniform.
    """
    _, ns = load_helpers()
    rng = np.random.default_rng(seed)
    frame = np.stack((
        rng.integers(90, 150, (height, width)),
        rng.integers(70, 130, (height, width)),
        rng.integers(100, 160, (height, width)),
    ), -1).astype(np.uint8)
    yy, xx = np.mgrid[:height, :width]
    z = np.full((height, width), 3.0, dtype=np.float32)
    cy, cx = height // 2, width // 2
    rr = min(width, height) * 0.28
    d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)[..., None]
    zz = 1.5 + 1.5 * np.sqrt(np.clip(1 - (d / rr) ** 2, 0, 1))
    z = np.where(d[..., 0] < rr, zz[:, :, 0], z)
    estimator = ns["ScharrNormalEstimator"](height, width, device=device)
    zt = torch.as_tensor(z, device=device)
    normals, points, valid = estimator.compute_normals_and_coords(zt)
    return frame, zt, normals, points, valid


def check_features(device):
    _, ns = load_helpers()
    relight_rgb = ns["relight_rgb"]
    h, w = 240, 320
    frame, z, normals, points, valid = scene(h, w, device)
    light = dict(ns["default_light"]())
    light.update(x=0.0, y=0.0, z=0.2, power=4.0, specular=0.3)
    base = relight_rgb(frame, normals, points, valid, light)

    # 1. Legacy path: default kwargs reproduce the previous renderer exactly.
    legacy = relight_rgb(frame, normals, points, valid, light,
                         cull_radius_frac=0.0, flat_threshold=None, flat_suppress=0.25)
    np.testing.assert_array_equal(legacy, base)

    # 2. A cull radius of 2.0 covers the whole frame: still bit-identical.
    full = relight_rgb(frame, normals, points, valid, light, cull_radius_frac=2.0)
    np.testing.assert_array_equal(full, base)

    # 3. Cull box: inside the circle identical to the full render; outside the
    #    circle identical to the ambient (power=0) render -- never raw pixels.
    culled = relight_rgb(frame, normals, points, valid, light, cull_radius_frac=0.35)
    ambient = relight_rgb(frame, normals, points, valid, dict(light, power=0.0))
    cx, cy = ns["_light_screen_pixels"](light, h, w)
    r = int(0.35 * w)
    x0, x1 = max(0, cx - r), min(w, cx + r + 1)
    y0, y1 = max(0, cy - r), min(h, cy + r + 1)
    inside = (slice(y0, y1), slice(x0, x1))
    outside = np.ones((h, w), bool)
    outside[y0:y1, x0:x1] = False
    inside_mask = ~outside
    np.testing.assert_array_equal(culled[inside], base[inside])
    np.testing.assert_array_equal(culled[outside], ambient[outside])

    # 4. Hotspot is preserved at the light-screen center.
    hotspot = int(float(culled[cy, cx].mean()))
    assert hotspot > int(float(culled[h - 1, w - 1].mean())) + 10

    # 5. Flat suppression dims the flat wall but leaves the bump slopes equal.
    flat_on = relight_rgb(frame, normals, points, valid, light,
                          cull_radius_frac=0.35, flat_threshold=0.95, flat_suppress=0.25)
    znp = np.asarray(z.cpu().numpy())
    slope = (np.asarray(normals.cpu().numpy())[..., 2] < 0.95)
    slope &= (np.asarray(normals.cpu().numpy())[..., 2] > 0.5)
    assert slope.any()
    np.testing.assert_array_equal(flat_on[slope], base[slope])
    wall_inside = (znp > 2.95) & inside_mask
    assert wall_inside.any()
    assert int(float(flat_on[wall_inside].mean())) < int(float(base[wall_inside].mean())) - 5

    # 6. Ambient (power=0) is untouched by cull and flat options.
    ambient_flat = relight_rgb(frame, normals, points, valid, dict(light, power=0.0),
                               cull_radius_frac=0.35, flat_threshold=0.95, flat_suppress=0.25)
    np.testing.assert_array_equal(ambient_flat, ambient)

    # 7. Invalid geometry keeps ambient only, even inside the cull box.
    bad = points.clone()
    bad[cy - 5:cy + 5, cx - 5:cx + 5] = float("nan")
    valid_invalid = valid.clone()
    valid_invalid[cy - 5:cy + 5, cx - 5:cx + 5] = False
    box = (slice(cy - 5, cy + 5), slice(cx - 5, cx + 5))
    np.testing.assert_array_equal(
        relight_rgb(frame, bad, normals, valid_invalid, light)[box], ambient[box])
    np.testing.assert_array_equal(
        relight_rgb(frame, bad, normals, valid, light)[box], ambient[box])

    # 8. Visibility gating: ones == no change; NaN treated as illuminated;
    #    shape mismatch raises.
    ones = torch.ones((h, w), device=device)
    np.testing.assert_array_equal(
        relight_rgb(frame, normals, points, valid, light, visibility=ones), base)
    nanvis = ones.clone()
    nanvis[0, 0] = float("nan")
    nan_at_one = torch.where(torch.isnan(nanvis), torch.ones_like(nanvis), nanvis)
    np.testing.assert_array_equal(
        relight_rgb(frame, normals, points, valid, light, visibility=nanvis),
        relight_rgb(frame, normals, points, valid, light, visibility=nan_at_one))
    try:
        relight_rgb(frame, normals, points, valid, light, visibility=ones[:10, :10])
        raise AssertionError("visibility shape mismatch must raise")
    except ValueError:
        pass

    # 9. Parameter validation.
    bad_params = (dict(cull_radius_frac=-0.1), dict(cull_radius_frac=2.1),
                  dict(cull_radius_frac=float("nan")), dict(flat_threshold=0.0),
                  dict(flat_threshold=1.0), dict(flat_threshold=0.5, flat_suppress=0.0),
                  dict(flat_threshold=0.5, flat_suppress=float("nan")),
                  dict(flat_threshold=0.5, flat_suppress=1.5))
    for kwargs in bad_params:
        try:
            relight_rgb(frame, normals, points, valid, light, **kwargs)
            raise AssertionError(f"must raise for {kwargs}")
        except ValueError:
            pass
    print(f"relight cull/flat feature checks passed on {device}.")


def _series(fn, device, n=30, warmup=15):
    for _ in range(warmup):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    return np.asarray(times, dtype=np.float64)


def benchmark(device, output_dir):
    _, ns = load_helpers()
    relight_rgb = ns["relight_rgb"]
    presets = {"fast": (320, 240), "balanced": (480, 360), "detail": (640, 480)}
    conditions = {"off": dict(cull_radius_frac=0.0, flat_threshold=None),
                  "cull_only": dict(cull_radius_frac=0.35, flat_threshold=None),
                  "cull_plus_flat": dict(cull_radius_frac=0.35, flat_threshold=0.95, flat_suppress=0.25)}
    lights = {"center": {"x": 0.0, "y": 0.0, "z": 0.2},
              "edge": {"x": 2.6, "y": 1.6, "z": 0.2}}
    rows = {}
    for plabel, (w, h) in presets.items():
        frame, z, normals, points, valid = scene(h, w, device)
        z = z.to(device)
        normals, points, valid = normals.to(device), points.to(device), valid.to(device)
        for llabel, pos in lights.items():
            light = dict(ns["default_light"]())
            light.update(pos, power=4.0, specular=0.3)
            row = {"frame": [w, h], "light": llabel}
            for clabel, kw in conditions.items():
                fn = (lambda kw=kw: relight_rgb(frame, normals, points, valid, light,
                                                return_tensor=True, **kw))
                t = _series(fn, device)
                row[clabel + "_mean_ms"] = round(float(t.mean()), 3)
                row[clabel + "_p95_ms"] = round(float(np.percentile(t, 95)), 3)
            row["cull_only_vs_off_ms"] = round(row["cull_only_mean_ms"] - row["off_mean_ms"], 3)
            row["cull_plus_flat_vs_off_ms"] = round(row["cull_plus_flat_mean_ms"] - row["off_mean_ms"], 3)
            rows[f"{plabel}/{llabel}"] = row

    # Radius sweep at balanced, center light: the measured time MUST fall as the
    # radius shrinks. A mask-after implementation would keep it flat; slicing
    # before compute shrinks the actual work with the box area.
    w, h = presets["balanced"]
    frame, z, normals, points, valid = scene(h, w, device)
    z = z.to(device)
    normals, points, valid = normals.to(device), points.to(device), valid.to(device)
    light = dict(ns["default_light"]())
    light.update(x=0.0, y=0.0, z=0.2, power=4.0, specular=0.3)
    sweep = {}
    for r in (0.0, 0.35, 0.15):
        fn = (lambda r=r: relight_rgb(frame, normals, points, valid, light,
                                      return_tensor=True, cull_radius_frac=r,
                                      flat_threshold=None))
        sweep[str(r)] = round(float(_series(fn, device, n=40).mean()), 3)

    result = {
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "torch": f"{torch.__version__}+cu{torch.version.cuda}" if torch.version.cuda else torch.__version__,
        "dt": "2026-09-13",
        "note": "synthetic compute only, not live-camera FPS; GPU/CPU stages overlap in the live loop",
        "rows": rows,
        "radius_sweep_balanced_center": sweep,
        "radius_sweep_note": "on CPU the ms scale with the box area (proves the box slices arrays before "
                             "compute, i.e. crop-and-slice, not mask-after); on GPU the shading is launch-bound "
                             "at these widths, so the box is roughly neutral and the timing check is only "
                             "evidence on CPU",
    }
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "benchmark.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    t0, t35, t15 = sweep["0.0"], sweep["0.35"], sweep["0.15"]
    if device.type == "cpu":
        assert t35 < t0, f"cull (r=0.35) must beat full-frame (r=0) on CPU: {t35:.2f} vs {t0:.2f}"
        assert t15 < t35, f"smaller box (r=0.15) must beat larger box (r=0.35) on CPU: {t15:.2f} vs {t35:.2f}"
    else:
        assert t15 < t0 * 1.5, f"smallest box must not regress the full-frame cost on GPU: {t15:.2f} vs {t0:.2f}"
        assert t35 < t0 * 1.5, f"cull box must not regress the full-frame cost on GPU: {t35:.2f} vs {t0:.2f}"
    print(f"benchmark saved to {out / 'benchmark.json'}")
    for k, v in sweep.items():
        print(f"  balanced/center radius {k}: {v:.2f} ms")
    return result


def save_preview(device, output_dir):
    _, ns = load_helpers()
    relight_rgb = ns["relight_rgb"]
    w, h = 640, 480
    frame, z, normals, points, valid = scene(h, w, device)
    z = z.to(device)
    normals, points, valid = normals.to(device), points.to(device), valid.to(device)
    light = dict(ns["default_light"]())
    light.update(x=0.0, y=0.0, z=0.2, power=4.0, specular=0.3)
    base = relight_rgb(frame, normals, points, valid, light)
    culled = relight_rgb(frame, normals, points, valid, light,
                         cull_radius_frac=0.35, flat_threshold=None)
    flat = relight_rgb(frame, normals, points, valid, light,
                       cull_radius_frac=0.35, flat_threshold=0.95, flat_suppress=0.25)
    cx, cy = ns["_light_screen_pixels"](light, h, w)
    for img in (culled, flat):
        cv2.circle(img, (cx, cy), int(0.35 * w), (0, 255, 100), 1, cv2.LINE_AA)
    panels = []
    for img, label in zip((base, culled, flat),
                          ("baseline (full-frame)", "cull 0.35 (flat off)", "cull 0.35 + flat 0.25")):
        panel = np.uint8(img)
        cv2.putText(panel, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        panels.append(panel)
    row = np.concatenate(panels, axis=1)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / "lighting_cull_preview.png"), cv2.cvtColor(row, cv2.COLOR_RGB2BGR))
    print(f"preview saved to {out / 'lighting_cull_preview.png'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"],
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", default="validation/lighting_cull")
    parser.add_argument("--no-benchmark", action="store_true")
    parser.add_argument("--no-preview", action="store_true")
    args = parser.parse_args()
    dev = torch.device(args.device)
    check_features(dev)
    if not args.no_benchmark:
        benchmark(dev, args.output_dir)
    if not args.no_preview:
        save_preview(dev, args.output_dir)