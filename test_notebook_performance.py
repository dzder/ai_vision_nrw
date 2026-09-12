"""Validate the Colab fast path without requiring a browser or remote runtime.

Optional --model-smoke uses only cached weights. --benchmark reports local
compute costs, not hosted Colab FPS or browser/network performance.
"""
import argparse
import ast
from collections import deque
import io
import json
from pathlib import Path
import time
from types import SimpleNamespace

import cv2
import numpy as np
import torch
from PIL import Image


def load_helpers():
    nb = json.loads(Path(__file__).with_name('test_l1.ipynb').read_text(encoding='utf-8'))
    ns = dict(time=time, deque=deque, Image=Image,
              DisplayImage=lambda **kw: SimpleNamespace(**kw), JPEG_QUALITY=75)
    for c in nb['cells']:
        tags = c['metadata'].get('tags', [])
        if set(tags) & {'geometry', 'lighting'}:
            exec(''.join(c['source']), ns)
        elif 'live' in tags:
            tree = ast.parse(''.join(c['source']))
            tree.body = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))]
            exec(compile(tree, 'notebook live helpers', 'exec'), ns)
    return nb, ns


def check_device_mapper(ns, device):
    reference = ns['RelativeGeometryDepth']()
    fast = ns['DeviceRelativeGeometryDepth']()
    rng = np.random.default_rng(9)
    # Compare exact full-sample bounds and OpenCV bilateral on small images,
    # including constant predictions and a resolution change (EMA must reset).
    for shape in [(31, 47), (31, 47), (17, 23)]:
        for raw in [np.full(shape, 2, np.float32), rng.normal(size=shape).astype(np.float32)]:
            expected = reference(raw, temporal=True)
            actual = fast(torch.as_tensor(raw, device=device), temporal=True)
            assert actual.device.type == device and actual.dtype == torch.float32
            np.testing.assert_allclose(actual.cpu().numpy(), expected, rtol=4e-5, atol=4e-5)
    for bad in [torch.zeros((1, 3), device=device), torch.full((3, 3), float('nan'), device=device)]:
        try:
            fast(bad)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid model prediction accepted')
    # Larger images use sampled bounds. Verify positivity, finite normals and
    # a bounded difference on a smooth ramp with a genuine foreground edge.
    y, x = np.mgrid[:180, :320].astype(np.float32)
    raw = x / 320 + y / 360 + (x > 170).astype(np.float32)
    fast.reset()
    actual = fast(torch.as_tensor(raw, device=device))
    expected = ns['RelativeGeometryDepth']()(raw)
    assert torch.isfinite(actual).all() and actual.min() >= 1/1.2 - 1e-5 and actual.max() <= 5
    assert np.mean(np.abs(actual.cpu().numpy()-expected)) < 0.04
    estimator = ns['ScharrNormalEstimator'](*raw.shape, device=device)
    normals, points, valid = estimator.compute_normals_and_coords(actual)
    rgb = np.full((*raw.shape, 3), 128, np.uint8)
    tensor = ns['relight_rgb'](rgb, normals, points, valid, return_tensor=True)
    assert tensor.device.type == device and tensor.dtype == torch.uint8
    np.testing.assert_array_equal(tensor.cpu().numpy(), ns['relight_rgb'](rgb, normals, points, valid))
    timer = ns['StageTimings'](device)
    timer.begin('work')
    result = actual.sin().cos()
    timer.end('work')
    result.cpu().numpy()
    assert timer.resolve()['work'] >= 0
    print(f'Fast geometry/lighting and event timing checks passed on {device}.')


def check_jpeg(ns):
    rgb = np.zeros((48, 64, 3), np.uint8)
    rgb[:] = [220, 70, 25]
    image = ns['encode_live_image'](rgb)
    assert image.format == 'jpeg' and image.data.startswith(b'\xff\xd8')
    decoded = np.asarray(Image.open(io.BytesIO(image.data)).convert('RGB'))
    assert decoded.shape == rgb.shape
    np.testing.assert_allclose(decoded.astype(float), rgb, atol=3)
    print('JPEG encoding checks passed: valid bytes, dimensions, RGB colors.')


def check_model(nb, ns, benchmark=False):
    from functools import partial
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ns.update(DEVICE=device, DTYPE=torch.float16 if device.type == 'cuda' else torch.float32,
              INPUT_SIZE=224, MODEL_ID='depth-anything/Depth-Anything-V2-Small-hf', FAST_PREPROCESS=True,
              AutoImageProcessor=SimpleNamespace(from_pretrained=partial(AutoImageProcessor.from_pretrained, local_files_only=True)),
              AutoModelForDepthEstimation=SimpleNamespace(from_pretrained=partial(AutoModelForDepthEstimation.from_pretrained, local_files_only=True)))
    source = ''.join(next(c for c in nb['cells'] if 'model' in c['metadata'].get('tags', []))['source'])
    exec(source, ns)
    processor = ns['processor']
    # Check actual tensor sizes against the installed HF processor, including
    # landscape, portrait, square and an image smaller than the model target.
    for target in (196, 224, 294):
        processor.size = dict(height=target, width=target)
        for h, w in [(180, 320), (360, 480), (320, 180), (224, 224), (60, 80)]:
            frame = np.full((h, w, 3), [40, 130, 210], np.uint8)
            actual = ns['prepare_depth_inputs'](frame)['pixel_values']
            reference = processor(images=frame, return_tensors='pt')['pixel_values'].to(device, ns['DTYPE'])
            assert actual.shape == reference.shape, (target, (h, w), actual.shape, reference.shape)
            torch.testing.assert_close(actual, reference, atol=0.002, rtol=0)
    processor.size = dict(height=224, width=224)
    y, x = np.mgrid[:180, :320]
    frame = np.stack((x % 256, y % 256, (x+y) % 256), -1).astype(np.uint8)
    timer = ns['StageTimings'](device)
    raw = ns['infer_relative_inverse'](frame, return_tensor=True, timings=timer)
    mapper = ns['DeviceRelativeGeometryDepth']()
    estimator = ns['ScharrNormalEstimator'](*frame.shape[:2], device=device)
    normals, points, valid = estimator.compute_normals_and_coords(mapper(raw))
    relit = ns['relight_rgb'](frame, normals, points, valid, return_tensor=True).cpu().numpy()
    assert np.isfinite(relit).all() and relit.shape == frame.shape
    assert all(v >= 0 for v in timer.resolve().values())
    selected_mapper, backend = ns['select_geometry_mapper'](raw, estimator)
    assert backend in ('cpu', 'gpu')
    z, normals, points, valid = ns['compute_live_geometry'](raw, selected_mapper, estimator)
    assert z.device == raw.device and torch.isfinite(points).all()
    for forced in ('cpu', 'gpu'):
        ns['GEOMETRY_BACKEND'] = forced
        _, actual_backend = ns['select_geometry_mapper'](raw, estimator)
        assert forced == actual_backend
    ns['GEOMETRY_BACKEND'] = 'auto'
    ns['FAST_PREPROCESS'] = False
    reference = ns['infer_relative_inverse'](frame)
    assert reference.shape == raw.shape and np.isfinite(reference).all()
    ns['FAST_PREPROCESS'] = True
    print(f'Cached-model fast/reference smoke checks passed on {device}; preprocessing sizes match at 196/224/294.')
    if benchmark:
        def measured(fn, count=12):
            for _ in range(3):
                fn()
            if device.type == 'cuda':
                torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(count):
                fn()
            if device.type == 'cuda':
                torch.cuda.synchronize()
            return (time.perf_counter()-start)*1000/count
        # A local compute-only comparison; never report this as Colab FPS.
        y, x = np.mgrid[:360, :480]
        image = np.stack((x % 256, y % 256, (x+y) % 256), -1).astype(np.uint8)
        raw = torch.as_tensor((x/480+y/360).astype(np.float32), device=device)
        old_mapper, fast_mapper = ns['RelativeGeometryDepth'](), ns['DeviceRelativeGeometryDepth']()
        estimator = ns['ScharrNormalEstimator'](360, 480, device=device)
        def old_geometry():
            return estimator.compute_normals_and_coords(old_mapper(raw.cpu().numpy(), temporal=True))
        def new_geometry():
            return estimator.compute_normals_and_coords(fast_mapper(raw, temporal=True, validate=False), validate=False)
        print(f'LOCAL ONLY 480x360 geometry: reference {measured(old_geometry):.1f} ms; GPU path {measured(new_geometry):.1f} ms')
        for target in (196, 224, 294):
            processor.size = dict(height=target, width=target)
            print(f'LOCAL ONLY model target {target}: {measured(lambda: ns["infer_relative_inverse"](image, return_tensor=True), 6):.1f} ms')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-smoke', action='store_true')
    parser.add_argument('--benchmark', action='store_true')
    args = parser.parse_args()
    nb, ns = load_helpers()
    for device in ['cpu'] + (['cuda'] if torch.cuda.is_available() else []):
        check_device_mapper(ns, device)
    check_jpeg(ns)
    if args.model_smoke or args.benchmark:
        check_model(nb, ns, args.benchmark)
