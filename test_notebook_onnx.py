"""ONNX CUDA export/parity/timing and simulated live lifecycle checks.

Default: a tiny model for integration checks. --model-smoke: cached real depth model.
No camera/window is opened. Prints compute timings, never measured live-camera FPS.
"""
import argparse
import ast
import json
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch

from test_notebook_performance import load_helpers


def main(real_model=False):
    nb, ns = load_helpers()
    if not torch.cuda.is_available():
        raise RuntimeError('ONNX experiment integration tests require CUDA.')
    setup = next(c for c in nb['cells'] if 'setup' in c['metadata'].get('tags', []))
    exec(''.join(setup['source']), ns)
    if real_model:
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        ns.update(AutoImageProcessor=SimpleNamespace(from_pretrained=partial(
            AutoImageProcessor.from_pretrained, local_files_only=True)),
            AutoModelForDepthEstimation=SimpleNamespace(from_pretrained=partial(
                AutoModelForDepthEstimation.from_pretrained, local_files_only=True)))
        model_cell = next(c for c in nb['cells'] if 'model' in c['metadata'].get('tags', []))
        exec(''.join(model_cell['source']), ns)
    else:
        class TinyDepth(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 1, 1)
                with torch.no_grad():
                    self.conv.weight.fill_(0.2)
                    self.conv.bias.fill_(1.0)

            def forward(self, pixel_values):
                return SimpleNamespace(predicted_depth=self.conv(pixel_values)[:, 0])

        ns['model'] = TinyDepth().cuda().half().eval()
        ns['processor'] = SimpleNamespace(size={'height': 196, 'width': 196},
                                           keep_aspect_ratio=True, ensure_multiple_of=14,
                                           rescale_factor=1/255)
        ns['_image_mean'] = torch.zeros((1,3,1,1), device='cuda')
        ns['_image_std'] = torch.ones((1,3,1,1), device='cuda')
        model_cell = next(c for c in nb['cells'] if 'model' in c['metadata'].get('tags', []))
        tree = ast.parse(''.join(model_cell['source']))
        tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
        exec(compile(tree, 'model helpers', 'exec'), ns)

    original_model = ns['model']
    original_infer = ns['infer_relative_inverse']
    original_prepare = ns['prepare_depth_inputs']
    original_size = dict(ns['processor'].size)
    original_weight = next(original_model.parameters()).detach().clone()
    experiment = next(c for c in nb['cells'] if 'onnx-experiment' in c['metadata'].get('tags', []))
    tree = ast.parse(''.join(experiment['source']))
    tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    exec(compile(tree, 'onnx experiment', 'exec'), ns)
    y, x = np.mgrid[:240, :320]
    frame = np.stack((x % 256, y % 256, (x+y) % 256), -1).astype(np.uint8)
    lifecycle = dict(start=0, stop=0, frame=0, displayed=0)

    def start():
        lifecycle['start'] += 1
        return True

    def stop():
        lifecycle['stop'] += 1

    def get():
        lifecycle['frame'] += 1
        return frame, ns['default_light'](), dict(mode='gesture', status='Simulated hand',
                                                 frame_id=lifecycle['frame'], hand_ms=0.3)

    def show(picture):
        assert picture.shape == frame.shape
        lifecycle['displayed'] += 1
        return True

    ns.update(start_webcam=start, stop_webcam=stop, get_frame=get, show_local_frame=show,
              annotate_local_hand=lambda *args: None, GEOMETRY_BACKEND='cpu')
    before_results = set(Path('models/onnx_experiments').glob('*/results.json'))
    report = ns['run_onnx_experiment'](live=True, count=6, live_frames=2)
    for path in set(Path('models/onnx_experiments').glob('*/results.json')) - before_results:
        saved = json.loads(path.read_text())
        saved['simulated_live'] = saved.pop('live', {})
        saved['test_harness'] = 'Simulated capture, hand tracking and display. Only compute timings used real GPU work.'
        path.write_text(json.dumps(saved, indent=2))
    assert report['quality_guardrails_passed']
    assert report['profile_kernel_events']['CUDAExecutionProvider'] > 0
    assert report['input_shape'] == [1,3,140,196]
    assert lifecycle == dict(start=2, stop=2, frame=24, displayed=24), lifecycle
    for backend in ('pytorch', 'onnx'):
        assert report['depth_path'][backend]['mean_ms'] > 0
        assert report['live'][backend]['frames'] == 2
        assert report['live'][backend]['hand_modes'] == ['gesture']
    assert ns['model'] is original_model and not original_model.training
    assert ns['infer_relative_inverse'] is original_infer
    assert ns['prepare_depth_inputs'] is original_prepare
    assert ns['processor'].size == original_size
    torch.testing.assert_close(next(original_model.parameters()), original_weight, rtol=0, atol=0)
    print('PASS: export, GPU binding/provider, numerical parity, paired benchmarks, '
          'simulated live frame counts/cleanup, and baseline state preservation.')
    print('Live timings above used simulated capture/hand/display and are NOT camera FPS.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-smoke', action='store_true')
    args = parser.parse_args()
    main(args.model_smoke)
