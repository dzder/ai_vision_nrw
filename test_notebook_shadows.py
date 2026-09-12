"""Validate actual notebook shadow helpers without camera/GUI or downloads.

--benchmark: paired shadow-off/on compute timings on synthetic geometry.
--model-smoke: cached depth model through shadows and rendering, no downloads.
--output-dir: save a synthetic comparison PNG and, with --benchmark, timing JSON.
"""
import argparse
from functools import partial
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import torch

from test_notebook_performance import load_helpers


def scene(ns, device='cpu', h=120, w=160, blocker=True):
    est = ns['ScharrNormalEstimator'](h, w, fx=w*.875, fy=w*.875, device=device)
    z = torch.full((h, w), 3., device=device)
    if blocker:
        z[round(h*.375):round(h*.625), round(w*.425):round(w*.575)] = 1.5
    normals, points, valid = est.compute_normals_and_coords(z)
    return est, z, normals, points, valid


def check_geometry(ns, device):
    shadow = ns['compute_shadow_visibility']
    light = dict(ns['default_light'](), x=-.6, y=0., z=0.)
    est, z, n, p, valid = scene(ns, device, blocker=False)
    assert est.intrinsics == (est.fx, est.fy, est.cx, est.cy)
    # No false occlusion on a plane, including angled rays and offscreen lights.
    for slope in (0., .15, -.3):
        plane_z = 2. / (1 - slope * est.ray_grid[..., 0])
        nn, pp, vv = est.compute_normals_and_coords(plane_z)
        for x, y, lz in ((-.6, 0, 0), (3., -2., -2.), (0., 0., .7)):
            actual = shadow(plane_z, nn, vv, dict(light, x=x, y=y, z=lz), est.intrinsics)
            assert actual.dtype == torch.float32 and actual.device.type == device
            assert torch.isfinite(actual).all()
            assert float(actual[3:-3, 3:-3].min()) > .99, (device, slope, x, float(actual.mean()))

    est, z, n, p, valid = scene(ns, device)
    actual = shadow(z, n, valid, light, est.intrinsics, filter_edges=False)
    # Analytic ray/foreground-plane intersection, independent of tracing code.
    fraction = (1.5-p[..., 2]) / (light['z']-p[..., 2])
    lamp = p.new_tensor([light[k] for k in ('x', 'y', 'z')])
    intersection = p + fraction[..., None]*(lamp-p)
    projected_x = est.fx*intersection[..., 0]/1.5+est.cx
    projected_y = est.fy*intersection[..., 1]/1.5+est.cy
    expected = ((projected_x >= 67.5) & (projected_x < 91.5) &
                (projected_y >= 44.5) & (projected_y < 74.5) & (z > 2))
    found = (actual < .5) & (z > 2)
    iou = (found & expected).sum() / (found | expected).sum()
    assert float(iou) > .8, ('analytic shadow IoU', float(iou))
    assert float(actual[z < 2].mean()) > .98  # Foreground does not shadow itself.

    def centroid(lamp):
        v = shadow(z, n, valid, lamp, est.intrinsics)
        weights = (1-v) * (z > 2)
        yy, xx = torch.meshgrid(torch.arange(z.shape[0], device=device),
                                torch.arange(z.shape[1], device=device), indexing='ij')
        assert weights.sum() > 50
        return float((weights*xx).sum()/weights.sum()), float((weights*yy).sum()/weights.sum())

    right = centroid(light)
    left = centroid(dict(light, x=.6))
    assert right[0] > 95 and left[0] < 65
    down = centroid(dict(light, x=0., y=-.5))
    up = centroid(dict(light, x=0., y=.5))
    assert down[1] > 70 and up[1] < 50
    far = centroid(dict(light, z=-1.))
    assert right[0] > far[0]+5  # Moving light away reduces lateral displacement.
    sequence = [centroid(dict(light, x=x))[0] for x in np.linspace(-.7, -.3, 7)]
    assert all(0 < a-b < 5 for a, b in zip(sequence, sequence[1:])), sequence

    for batch in (1, 7, 32):
        other = shadow(z, n, valid, light, est.intrinsics, batch_size=batch, filter_edges=False)
        torch.testing.assert_close(actual, other, rtol=0, atol=0)
    # Invalid occluders must not cast a shadow; invalid receiver returns lit.
    invalid = valid & (z > 2)
    cleaned = shadow(z, n, invalid, light, est.intrinsics)
    assert float(cleaned.min()) > .99
    broken_z, broken_n = z.clone(), n.clone()
    broken_z[10:20, 10:20] = float('nan')
    broken_z[20:30, 10:20] = 0
    broken_n[30:40, 10:20] = float('nan')
    broken = shadow(broken_z, broken_n, valid, light, est.intrinsics)
    assert torch.isfinite(broken).all() and ((broken >= 0) & (broken <= 1)).all()
    assert (broken[10:40, 10:20] == 1).all()
    assert (shadow(z, n, torch.zeros_like(valid), light, est.intrinsics) == 1).all()
    for xyz in ((0, 0, 0), (0, 0, -2), (30, -20, -2), (0, 0, 3), (0, 0, 1.5)):
        result = shadow(z, n, valid, dict(light, **dict(zip(('x', 'y', 'z'), xyz))), est.intrinsics)
        assert torch.isfinite(result).all()
    # Actual non-default intrinsics, camera shapes, upsampling and quality preset.
    for h, w in ((48, 64), (180, 320), (240, 320), (360, 480), (480, 640), (120, 80)):
        e, zz, nn, pp, vv = scene(ns, device, h=h, w=w)
        output = shadow(zz, nn, vv, light, e.intrinsics)
        assert output.shape == zz.shape and torch.isfinite(output).all()
        assert float(output[zz < 2].mean()) > .97
        assert ((output > 0) & (output < 1)).any()  # Filter actually softens edges.
    quality = shadow(z, n, valid, light, est.intrinsics, preset='quality')
    assert ((quality < .5) & expected).sum() > expected.sum()*.8
    print(f'Shadow geometry passed on {device}; analytic silhouette IoU={float(iou):.3f}.')
    return shadow(z, n, valid, light, est.intrinsics).cpu()


def check_rendering(ns, device):
    est, z, n, p, valid = scene(ns, device)
    rgb = np.full((*z.shape, 3), [90, 120, 150], np.uint8)
    light = dict(ns['default_light'](), x=-.6, y=0.)
    render = ns['relight_rgb']
    original = render(rgb, n, p, valid, light)
    np.testing.assert_array_equal(original, render(rgb, n, p, valid, light, visibility=torch.ones_like(z)))
    ambient = render(rgb, n, p, valid, dict(light, power=0))
    np.testing.assert_array_equal(ambient, render(rgb, n, p, valid, light, visibility=torch.zeros_like(z)))
    visibility = ns['compute_shadow_visibility'](z, n, valid, light, est.intrinsics)
    shadowed = render(rgb, n, p, valid, light, visibility=visibility)
    assert np.all(shadowed <= original) and np.any(shadowed < original)
    # Contract errors and nonfinite masks must not poison rendering.
    try:
        render(rgb, n, p, valid, light, visibility=torch.ones(2, 2))
    except ValueError:
        pass
    else:
        raise AssertionError('Mismatched visibility accepted')
    np.testing.assert_array_equal(original, render(rgb, n, p, valid, light, visibility=torch.full_like(z, float('nan'))))
    real = ns['compute_shadow_visibility']
    def must_skip(*a, **kw):
        raise AssertionError('Disabled/zero-power path traced rays')
    ns['compute_shadow_visibility'] = must_skip
    try:
        ns['SHADOWS_ENABLED'] = False
        off = ns['configured_shadow_visibility'](z, n, valid, light, est.intrinsics)
        np.testing.assert_array_equal(original, render(rgb, n, p, valid, light, visibility=off))
        ns['SHADOWS_ENABLED'] = True
        ns['configured_shadow_visibility'](z, n, valid, dict(light, power=0), est.intrinsics)
    finally:
        ns['compute_shadow_visibility'] = real
        ns.pop('SHADOWS_ENABLED', None)
    print(f'Shadow rendering passed on {device}: diffuse/specular, ambient, identity, disabled/zero-power bypass.')


def benchmark(ns):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    from datetime import datetime, timezone
    report = dict(device=device, gpu=torch.cuda.get_device_name() if device == 'cuda' else None,
                  recorded_at=datetime.now(timezone.utc).isoformat(), torch=torch.__version__,
                  shadow_preset='fast', bias=.01, thickness=.03, warmup_pairs=15, measured_pairs=30,
                  method='alternating off/on order; wall time with device synchronization; no image download',
                  source='synthetic geometry; excludes depth, camera, hand inference, UI and display', presets={})
    for name, w, h in (('fast', 320, 240), ('balanced', 480, 360), ('detail', 640, 480)):
        est, z, n, p, valid = scene(ns, device, h, w)
        rgb = np.full((h, w, 3), 120, np.uint8)
        light = dict(ns['default_light'](), x=-.6, y=0)
        samples = {False: [], True: []}
        def run(enabled):
            visibility = ns['compute_shadow_visibility'](z, n, valid, light, est.intrinsics) if enabled else None
            return ns['relight_rgb'](rgb, n, p, valid, light, return_tensor=True, visibility=visibility)
        for i in range(45):
            for enabled in ((False, True) if i % 2 == 0 else (True, False)):
                if device == 'cuda':
                    torch.cuda.synchronize()
                start = time.perf_counter()
                result = run(enabled)
                if device == 'cuda':
                    torch.cuda.synchronize()
                elapsed = (time.perf_counter()-start)*1000
                if i >= 15:
                    samples[enabled].append(elapsed)
        report['presets'][name] = dict(width=w, height=h,
            off_mean_ms=float(np.mean(samples[False])), on_mean_ms=float(np.mean(samples[True])),
            added_mean_ms=float(np.mean(samples[True])-np.mean(samples[False])),
            off_p95_ms=float(np.percentile(samples[False], 95)), on_p95_ms=float(np.percentile(samples[True], 95)))
    print(json.dumps(report, indent=2))
    return report


def check_photo(nb, ns):
    """Execute the actual photo cell with a synthetic image and a fake plot UI."""
    from PIL import Image
    photo_source = ''.join(next(c for c in nb['cells'] if 'photo' in c['metadata'].get('tags', []))['source'])
    photo_source = photo_source.replace('PHOTO_PATH = None', 'PHOTO_PATH = "synthetic"', 1)
    frame = np.full((48, 64, 3), 120, np.uint8)
    shown = []
    axes = [SimpleNamespace(imshow=lambda image, **kw: shown.append(np.asarray(image).copy()),
                            set_title=lambda *a: None, axis=lambda *a: None) for _ in range(4)]
    fig = SimpleNamespace(colorbar=lambda *a, **kw: None)
    local = dict(ns, DEVICE='cpu', ASSUMED_HFOV_DEG=60.,
                 Image=SimpleNamespace(open=lambda path: Image.fromarray(frame)),
                 infer_relative_inverse=lambda f: np.ones(f.shape[:2], np.float32),
                 plt=SimpleNamespace(subplots=lambda *a, **kw: (fig, axes),
                                     tight_layout=lambda: None, show=lambda: None))
    for enabled, visibility_view in ((True, False), (False, False), (True, True)):
        local.update(SHADOWS_ENABLED=enabled, SHOW_SHADOW_VISIBILITY=visibility_view)
        shown.clear()
        exec(compile(photo_source, 'actual notebook photo cell', 'exec'), local)
        assert len(shown) == 4
        np.testing.assert_array_equal(shown[0], frame)
        expected = local['visibility'].cpu().numpy() if visibility_view else local['relit']
        np.testing.assert_array_equal(shown[3], expected)
    print('Photo cell passed: actual helper integration, relit/visibility plots, and disabled shadows.')


def save_preview(ns, directory):
    import cv2
    est, z, n, p, valid = scene(ns, h=240, w=320)
    rgb = np.full((240, 320, 3), [145, 157, 166], np.uint8)
    rows = []
    for x in (-.6, .6):
        light = dict(ns['default_light'](), x=x, y=0., z=0.)
        v = ns['compute_shadow_visibility'](z, n, valid, light, est.intrinsics)
        off = ns['relight_rgb'](rgb, n, p, valid, light)
        on = ns['relight_rgb'](rgb, n, p, valid, light, visibility=v)
        mask = np.repeat(np.rint(v.cpu().numpy()[..., None]*255).astype(np.uint8), 3, axis=-1)
        panels = []
        for label, picture in ((f'Light X={x:+.1f}: shadows off', off), ('Shadows on', on), ('Visibility: white = lit', mask)):
            panel = np.concatenate((np.full((30, 320, 3), 24, np.uint8), picture), axis=0)
            cv2.putText(panel, label, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, .5, (235, 235, 235), 1, cv2.LINE_AA)
            panels.append(panel)
        rows.append(np.concatenate(panels, axis=1))
    target = directory / 'shadow_comparison.png'
    if not cv2.imwrite(str(target), cv2.cvtColor(np.concatenate(rows, axis=0), cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f'Failed to save {target}')
    print(f'Synthetic visual comparison saved: {target}')


def model_smoke(nb, ns):
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ns.update(DEVICE=device, DTYPE=torch.float16 if device.type == 'cuda' else torch.float32,
        INPUT_SIZE=196, MODEL_ID='depth-anything/Depth-Anything-V2-Small-hf', FAST_PREPROCESS=True,
        AutoImageProcessor=SimpleNamespace(from_pretrained=partial(AutoImageProcessor.from_pretrained, local_files_only=True)),
        AutoModelForDepthEstimation=SimpleNamespace(from_pretrained=partial(AutoModelForDepthEstimation.from_pretrained, local_files_only=True)))
    source = ''.join(next(c for c in nb['cells'] if 'model' in c['metadata'].get('tags', []))['source'])
    exec(source, ns)
    rng = np.random.default_rng(4)
    frame = rng.integers(0, 256, (240, 320, 3), dtype=np.uint8)
    raw = ns['infer_relative_inverse'](frame, return_tensor=True)
    est = ns['ScharrNormalEstimator'](*raw.shape, device=device)
    mapper = ns['DeviceRelativeGeometryDepth']() if device.type == 'cuda' else ns['RelativeGeometryDepth']()
    z, n, p, valid = ns['compute_live_geometry'](raw, mapper, est)
    visibility = ns['compute_shadow_visibility'](z, n, valid, ns['default_light'](), est.intrinsics)
    result = ns['relight_rgb'](frame, n, p, valid, visibility=visibility)
    assert result.shape == frame.shape and result.dtype == np.uint8
    assert torch.isfinite(visibility).all()
    print('Cached model -> geometry -> shadow -> relighting smoke passed. No live camera/FPS claim.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--model-smoke', action='store_true')
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    nb, ns = load_helpers()
    cpu = check_geometry(ns, 'cpu')
    check_rendering(ns, 'cpu')
    if torch.cuda.is_available():
        gpu = check_geometry(ns, 'cuda')
        torch.testing.assert_close(cpu, gpu, atol=1e-4, rtol=1e-4)
        check_rendering(ns, 'cuda')
        print('CPU/CUDA shadow agreement passed.')
    check_photo(nb, ns)
    if args.model_smoke:
        model_smoke(nb, ns)
    if args.benchmark:
        report = benchmark(ns)
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        if args.benchmark:
            (args.output_dir / 'benchmark.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
        save_preview(ns, args.output_dir)


if __name__ == '__main__':
    main()
