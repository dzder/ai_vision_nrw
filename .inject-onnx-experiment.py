import ast
import json
from pathlib import Path

path = Path('test_l1.ipynb')
original_bytes = path.read_bytes()
original_text = original_bytes.decode('utf-8')
original = json.loads(original_text)
source = Path('.onnx-experiment-cell.py').read_text(encoding='utf-8')
ast.parse(source)
if any('onnx-experiment' in c.get('metadata', {}).get('tags', []) for c in original['cells']):
    pos = original_text.index('[', original_text.index('"cells"')) + 1
    for cell in original['cells']:
        while original_text[pos] in ' \r\n\t,':
            pos += 1
        _, end = json.JSONDecoder().raw_decode(original_text, pos)
        if 'onnx-experiment' in cell['metadata'].get('tags', []):
            cell['source'] = source.splitlines(True)
            replacement = json.dumps(cell, ensure_ascii=False, indent=1).replace('\n', '\n  ')
            updated_text = original_text[:pos] + replacement + original_text[end:]
            updated = json.loads(updated_text)
            assert updated['cells'][:13] == json.loads(original_text)['cells'][:13]
            path.write_bytes(updated_text.encode('utf-8'))
            print('Updated only the added experiment cell; original cells and outputs unchanged.')
            raise SystemExit(0)
        pos = end
notes = '''## Optional ONNX experiment — original results preserved

Run the earlier setup, geometry, lighting and model cells first. This separate experiment
exports the **current preset's actual tensor dimensions**, verifies CUDA execution, binds
GPU inputs/outputs, checks depth/normal/relighting differences, and compares warmed-up
PyTorch FP16 and ONNX timings on identical inputs in alternating order.

The default uses synthetic images and reports **compute throughput, not live camera FPS**.
Set `ONNX_IMAGE_PATH` to a scene photo for a more useful visual comparison. The quality
image saved alongside the JSON is **RGB | PyTorch relighting | ONNX relighting**.
Numerical thresholds are experiment guardrails; inspect real faces, hands and edges too.

For a real FPS comparison, run and stop the original local live cell once (to define its
camera/gesture helpers), then set **`ONNX_LIVE_COMPARE = True`** below. It runs 150 measured
frames per backend after 10 warmups, with your current hand and diagnostic settings.
**Q/Esc cancels**; G/C/R and sliders work as before. The two live runs observe different
camera frames, so maintain the same scene and hand mode. Camera/display and hand tracking
are included in live FPS. Missing hand tracking is reported as manual, not hidden.

This cell never replaces the original inference function or its saved outputs. Each run
writes a fresh `models/onnx_experiments/<timestamp>/` directory with the export, provider
profile, quality image and measurements. Export/session startup are excluded from timing.
If CUDA or quality checks fail, the experiment stops and the original backend remains usable.
Rerun for each preset; a camera aspect-ratio change requires a matching export.

One-time dependencies in the notebook kernel (CUDA 12/cuDNN 9, including this MX550 setup):
`%pip install onnx==1.17.0 onnxruntime-gpu==1.20.2 "protobuf<5"`
Keep the installed CUDA PyTorch build. No automatic package installation occurs in this cell.

References: [CUDA compatibility and streams](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html),
[GPU I/O binding](https://onnxruntime.ai/docs/performance/tune-performance/iobinding.html).
'''
cells = [dict(cell_type='markdown', metadata={'tags': ['onnx-experiment-notes']}, source=notes.splitlines(True)),
         dict(cell_type='code', execution_count=None, metadata={'tags': ['onnx-experiment']},
              outputs=[], source=source.splitlines(True))]
# Insert before the cells array's closing bracket; existing bytes and outputs stay intact.
start = original_text.index('[', original_text.index('"cells"'))
_, end = json.JSONDecoder().raw_decode(original_text, start)
insertion = ',\n' + ',\n'.join('  '+json.dumps(c, ensure_ascii=False, indent=1).replace('\n', '\n  ') for c in cells) + '\n '
updated_text = original_text[:end-1].rstrip() + insertion + original_text[end-1:]
updated = json.loads(updated_text)
assert updated['cells'][:-2] == original['cells']
assert {k:v for k,v in updated.items() if k != 'cells'} == {k:v for k,v in original.items() if k != 'cells'}
path.write_bytes(updated_text.encode('utf-8'))
print(f'Added two cells; all {len(original["cells"])} original cells, metadata and outputs are identical.')
