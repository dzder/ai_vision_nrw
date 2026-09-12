# Optional, isolated ONNX experiment. Run setup/geometry/lighting/model cells first.
# For live A/B, run the existing live cell once, stop with Q, then enable below.
# Missing dependencies: %pip install onnx==1.17.0 onnxruntime-gpu==1.20.2 "protobuf<5"
ONNX_LIVE_COMPARE = False  # True: 150 frames of PyTorch, then 150 of ONNX; Q stops.
ONNX_IMAGE_PATH = None  # Optional real scene photo; None uses labeled synthetic inputs.
ONNX_BENCHMARK_FRAMES = 30
ONNX_LIVE_FRAMES = 150


def run_onnx_experiment(live=False, image_path=None, count=30, live_frames=150):
    """Export + paired CUDA timings; preserve the baseline model/functions/settings.

    Results are compute throughput unless explicitly labeled live completed-loop FPS.
    Each invocation writes a separate models/onnx_experiments/<timestamp> directory.
    """
    import copy
    import datetime
    import json
    import time
    from pathlib import Path
    import numpy as np
    import torch  # Load PyTorch CUDA/cuDNN DLLs before importing ONNX Runtime.
    import torch.nn.functional as F
    try:
        import onnx
        import onnxruntime as ort
    except ImportError as error:
        raise RuntimeError('Install onnx==1.17.0 onnxruntime-gpu==1.20.2 "protobuf<5" in this kernel.') from error

    required = ('model', 'processor', 'prepare_depth_inputs', 'infer_relative_inverse',
                'ScharrNormalEstimator', 'RelativeGeometryDepth', 'DeviceRelativeGeometryDepth', 'relight_rgb')
    missing = [name for name in required if name not in globals()]
    if missing:
        raise RuntimeError(f'Run the earlier setup, geometry, lighting and model cells first: {missing}')
    if DEVICE.type != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('This experiment requires the CUDA notebook kernel; baseline is unchanged.')
    if not str(torch.version.cuda).startswith('12.') or torch.backends.cudnn.version() < 90000:
        raise RuntimeError('This pinned ONNX Runtime build expects CUDA 12 and cuDNN 9.')
    if model.training:
        raise RuntimeError('Run the model setup cell first; the baseline must already be in eval mode.')
    if 'CUDAExecutionProvider' not in ort.get_available_providers():
        raise RuntimeError('CUDAExecutionProvider unavailable. Install onnxruntime-gpu, not the CPU package.')
    if count < 2 or live_frames < 1:
        raise ValueError('Use at least 2 benchmark frames and 1 live frame.')
    if live and (globals().get('LIVE_BACKEND') != 'local' or 'start_webcam' not in globals()):
        raise RuntimeError('For live A/B, run and stop the earlier local live cell first.')
    if globals().get('_local_camera') is not None:
        raise RuntimeError('Stop the existing camera loop with Q before starting this experiment.')

    width = int(CAPTURE_WIDTH)
    height = max(3, round(width * 0.75))
    if image_path:
        bgr = cv2.imread(str(image_path))
        if bgr is None:
            raise ValueError(f'Cannot read image: {image_path}')
        height = max(3, round(width * bgr.shape[0] / bgr.shape[1]))
        scene = cv2.cvtColor(cv2.resize(bgr, (width, height)), cv2.COLOR_BGR2RGB)
        frames = [np.ascontiguousarray(np.roll(scene, shift, axis=1)) for shift in (0, 3, 7, 11)]
        source = 'photo and shifted variants (compute-only)'
    else:
        y, x = np.mgrid[:height, :width]
        scene = np.stack((x % 256, y % 256, (x + y) % 256), -1).astype(np.uint8)
        frames = [np.ascontiguousarray(np.roll(scene, shift, axis=1)) for shift in (0, 3, 7, 11)]
        source = 'synthetic gradients (compute-only; real-scene quality still needs inspection)'

    # Freeze only the experiment's preprocessing configuration, never mutate processor/model.
    import types
    prep_globals = dict(prepare_depth_inputs.__globals__)
    prep_globals['processor'] = copy.deepcopy(processor)
    prepare = types.FunctionType(prepare_depth_inputs.__code__, prep_globals,
                                 prepare_depth_inputs.__name__, prepare_depth_inputs.__defaults__)
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    stream = torch.cuda.current_stream(device)
    device_id = device.index if device.index is not None else torch.cuda.current_device()
    sample = prepare(frames[0])['pixel_values']
    fixed_shape = tuple(sample.shape)
    run_dir = Path('models/onnx_experiments') / datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    run_dir.mkdir(parents=True, exist_ok=False)
    report = dict(source=source, gpu=torch.cuda.get_device_name(device), torch=torch.__version__,
                  cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                  onnxruntime=ort.__version__, dtype=str(dtype), input_shape=list(fixed_shape),
                  frame_shape=list(frames[0].shape), fast_preprocess=bool(FAST_PREPROCESS))
    print(f'Experiment: {run_dir}\n{source}\nActual model input: {fixed_shape}; {dtype}')

    class ExportDepth(torch.nn.Module):
        def __init__(self, depth_model):
            super().__init__()
            self.depth_model = depth_model

        def forward(self, pixel_values):
            return self.depth_model(pixel_values=pixel_values).predicted_depth

    with torch.inference_mode():
        wrapper = ExportDepth(model).eval()
        output_shape = tuple(wrapper(sample).shape)
        model_path = run_dir / 'depth.onnx'
        print('Exporting fixed-shape depth model (one-time cost, excluded from timing)...')
        torch.onnx.export(wrapper, (sample,), str(model_path), input_names=['pixel_values'],
                          output_names=['predicted_depth'], opset_version=17,
                          do_constant_folding=True, dynamic_axes=None, dynamo=False)
    onnx.checker.check_model(str(model_path))
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.enable_profiling = True  # One warmup run only; steady-state profiling is disabled below.
    options.profile_file_prefix = str(run_dir / 'provider_profile')
    cuda_options = dict(device_id=device_id, user_compute_stream=str(stream.cuda_stream),
                        cudnn_conv_algo_search='HEURISTIC', cudnn_conv_use_max_workspace='0')
    session = ort.InferenceSession(str(model_path), sess_options=options,
                                   providers=[('CUDAExecutionProvider', cuda_options), 'CPUExecutionProvider'])
    session.disable_fallback()
    if session.get_providers()[0] != 'CUDAExecutionProvider':
        raise RuntimeError('ONNX CUDA initialization failed; refusing to benchmark a CPU fallback.')
    output_dtype = {'tensor(float16)': torch.float16, 'tensor(float)': torch.float32}[session.get_outputs()[0].type]
    bound_output = torch.empty(output_shape, device=device, dtype=output_dtype)
    binding = session.io_binding()
    binding.bind_output('predicted_depth', 'cuda', device_id,
                        np.float16 if output_dtype == torch.float16 else np.float32,
                        output_shape, bound_output.data_ptr())

    @torch.inference_mode()
    def predict_onnx(pixels):
        if tuple(pixels.shape) != fixed_shape:
            raise ValueError(f'Export expects {fixed_shape}, received {tuple(pixels.shape)}. '
                             'Match the camera aspect ratio or rerun with a matching ONNX_IMAGE_PATH.')
        if pixels.dtype != dtype or pixels.device != device or not pixels.is_contiguous():
            raise ValueError('ONNX input must be contiguous and match the export device/dtype.')
        if torch.cuda.current_stream(device) != stream:
            raise RuntimeError('Run ONNX on the CUDA stream used when creating this experiment.')
        binding.bind_input('pixel_values', 'cuda', device_id,
                           np.float16 if dtype == torch.float16 else np.float32,
                           fixed_shape, pixels.data_ptr())
        session.run_with_iobinding(binding)
        return bound_output  # Reused buffer; consume on the same stream before the next call.

    predict_onnx(sample)
    torch.cuda.synchronize(device)
    profile_path = session.end_profiling()
    events = json.loads(Path(profile_path).read_text())
    provider_counts = {}
    for event in events:
        provider = event.get('args', {}).get('provider')
        if provider:
            provider_counts[provider] = provider_counts.get(provider, 0) + 1
    if not provider_counts.get('CUDAExecutionProvider'):
        raise RuntimeError('Profile contains no CUDA kernels; stopping the experiment.')
    report['profile_kernel_events'] = provider_counts
    print(f'Verified executed providers: {provider_counts} (CPU shape operations may remain).')

    @torch.inference_mode()
    def infer(frame, backend):
        pixels = prepare(frame)['pixel_values']
        prediction = model(pixel_values=pixels).predicted_depth if backend == 'pytorch' else predict_onnx(pixels)
        return F.interpolate(prediction[:, None].float(), size=frame.shape[:2],
                             mode='bilinear', align_corners=False)[0, 0]

    # Same normalization/smoothing for both; independent state avoids contaminating the baseline.
    def geometry(raw, mapper, estimator):
        if isinstance(mapper, DeviceRelativeGeometryDepth):
            z = mapper(raw, temporal=True, validate=False)
        else:
            z = mapper(raw.cpu().numpy(), temporal=True)
        z = torch.as_tensor(z, device=device, dtype=torch.float32)
        normals, points, valid = estimator.compute_normals_and_coords(z, validate=False)
        return z, normals, points, valid

    quality = []
    pictures = None
    with torch.inference_mode():
        for frame in frames:
            baseline, candidate = infer(frame, 'pytorch'), infer(frame, 'onnx')
            if not torch.isfinite(candidate).all() or not torch.isfinite(baseline).all():
                raise RuntimeError('Non-finite depth detected; ONNX experiment stopped, baseline unchanged.')
            scale = (torch.quantile(baseline.flatten(), .98) - torch.quantile(baseline.flatten(), .02)).clamp_min(1e-6)
            nrmse = float(torch.mean((baseline-candidate)**2).sqrt() / scale)
            estimator = ScharrNormalEstimator(*frame.shape[:2], hfov_deg=ASSUMED_HFOV_DEG, device=device)
            a = geometry(baseline, RelativeGeometryDepth(), estimator)
            b = geometry(candidate, RelativeGeometryDepth(), estimator)
            mask = a[3] & b[3]
            if not mask.any():
                raise RuntimeError('No valid geometry for ONNX quality comparison.')
            angles = torch.rad2deg(torch.acos((a[1]*b[1]).sum(-1).clamp(-1, 1)))[mask]
            ra = relight_rgb(frame, a[1], a[2], a[3])
            rb = relight_rgb(frame, b[1], b[2], b[3])
            mae = float(np.abs(ra.astype(np.float32)-rb.astype(np.float32)).mean())
            quality.append(dict(depth_nrmse=nrmse, normal_mean_deg=float(angles.mean()),
                                normal_p95_deg=float(torch.quantile(angles, .95)), relit_mae_255=mae))
            if pictures is None:
                pictures = (frame, ra, rb)
    report['quality'] = quality
    print('Quality:', {key: round(max(q[key] for q in quality), 5) for key in quality[0]})
    # Experiment guardrails, not a guarantee of real-scene visual equivalence.
    quality_ok = all(q['depth_nrmse'] < .02 and q['normal_mean_deg'] < 3 and q['relit_mae_255'] < 3 for q in quality)
    report['quality_guardrails_passed'] = quality_ok
    cv2.imwrite(str(run_dir / 'quality_rgb_pytorch_onnx.jpg'),
                cv2.cvtColor(np.concatenate(pictures, axis=1), cv2.COLOR_RGB2BGR))
    display(DisplayImage(filename=str(run_dir / 'quality_rgb_pytorch_onnx.jpg')))
    if not quality_ok:
        (run_dir / 'results.json').write_text(json.dumps(report, indent=2))
        raise RuntimeError(f'ONNX quality guardrails failed. Inspect {run_dir}; original backend is unchanged.')

    # Fix the geometry choice across backends for a fair end-to-end compute comparison.
    choice = globals().get('GEOMETRY_BACKEND', 'auto')
    estimator = ScharrNormalEstimator(height, width, hfov_deg=ASSUMED_HFOV_DEG, device=device)
    if choice == 'auto':
        if 'select_geometry_mapper' in globals():
            _, choice = select_geometry_mapper(infer(frames[0], 'pytorch'), estimator)
        else:
            choice = 'cpu'
    factory = {'cpu': RelativeGeometryDepth, 'gpu': DeviceRelativeGeometryDepth}[choice]
    report['geometry_backend'] = choice

    def paired_times(functions):
        samples = {name: [] for name in functions}
        with torch.inference_mode():
            for i in range(10):
                for fn in functions.values():
                    fn(frames[i % len(frames)])
            torch.cuda.synchronize(device)
            for i in range(count):
                order = list(functions) if i % 2 == 0 else list(reversed(functions))
                for name in order:
                    torch.cuda.synchronize(device)
                    start = time.perf_counter()
                    functions[name](frames[i % len(frames)])
                    torch.cuda.synchronize(device)
                    samples[name].append((time.perf_counter()-start)*1000)
        return {name: dict(mean_ms=float(np.mean(values)), p50_ms=float(np.median(values)),
                           p95_ms=float(np.percentile(values, 95)), compute_fps=1000/float(np.mean(values)))
                for name, values in samples.items()}

    report['depth_path'] = paired_times({name: (lambda frame, n=name: infer(frame, n))
                                         for name in ('pytorch', 'onnx')})
    mappers = {name: factory() for name in ('pytorch', 'onnx')}

    def render_compute(frame, backend):
        raw = infer(frame, backend)
        z, normals, points, valid = geometry(raw, mappers[backend], estimator)
        return relight_rgb(frame, normals, points, valid, return_tensor=True).cpu().numpy()

    report['render_compute'] = paired_times({name: (lambda frame, n=name: render_compute(frame, n))
                                             for name in ('pytorch', 'onnx')})
    for stage in ('depth_path', 'render_compute'):
        a, b = report[stage]['pytorch'], report[stage]['onnx']
        print(f'{stage}: PyTorch {a["mean_ms"]:.2f} ms ({a["compute_fps"]:.2f}/s), '
              f'ONNX {b["mean_ms"]:.2f} ms ({b["compute_fps"]:.2f}/s); '
              f'{a["mean_ms"]/b["mean_ms"]:.2f}x speedup')
    print('Compute rates exclude camera, hand tracking, UI and display. They are not live FPS.')
    (run_dir / 'results.json').write_text(json.dumps(report, indent=2))

    if live:
        # Explicit backend argument: never replace infer_relative_inverse or run_live.
        report['live'] = {}
        for backend in ('pytorch', 'onnx'):
            durations, hand_modes = [], set()
            mapper, live_estimator = factory(), None
            print(f'Live {backend.upper()}: {live_frames} measured frames after 10 warmups. '
                  'Same settings; separate camera runs. Q/Esc cancels the comparison.')
            cancelled = False
            try:
                if not start_webcam():
                    break
                with torch.inference_mode():
                    for i in range(live_frames + 10):
                        start = time.perf_counter()
                        packet = get_frame()  # Includes actual same-frame MediaPipe tracking.
                        if packet is None:
                            cancelled = True
                            break
                        frame, light, hand = packet
                        h, w = frame.shape[:2]
                        if live_estimator is None:
                            if tuple(prepare(frame)['pixel_values'].shape) != fixed_shape:
                                raise ValueError('Camera aspect ratio differs from export; use a matching scene photo to export.')
                            # Validate a real camera frame too; startup is outside measured FPS.
                            reference, candidate = infer(frame, 'pytorch'), infer(frame, 'onnx')
                            scale = (torch.quantile(reference.flatten(), .98)-torch.quantile(reference.flatten(), .02)).clamp_min(1e-6)
                            if (not torch.isfinite(candidate).all() or not torch.isfinite(reference).all()
                                    or float((reference-candidate).square().mean().sqrt()/scale) >= .02):
                                raise RuntimeError('Live-frame ONNX depth parity failed; stopping the experiment.')
                            live_estimator = ScharrNormalEstimator(h, w, hfov_deg=ASSUMED_HFOV_DEG, device=device)
                        elif (live_estimator.H, live_estimator.W) != (h, w):
                            raise RuntimeError('Camera resolution changed; rerun the experiment.')
                        raw = infer(frame, backend)
                        z, normals, points, valid = geometry(raw, mapper, live_estimator)
                        relit = relight_rgb(frame, normals, points, valid, light, return_tensor=True)
                        if SHOW_DIAGNOSTICS:
                            nrgb = torch.where(valid[..., None], ((normals+1)*127.5).round().clamp(0,255), 0).to(torch.uint8)
                            picture = np.asarray(live_panel(frame, z.cpu().numpy(), nrgb.cpu().numpy(), relit.cpu().numpy())).copy()
                        else:
                            picture = relit.cpu().numpy().copy()
                        annotate_local_hand(picture, hand, h, w)
                        rate = f'{len(durations)/sum(durations):.1f} FPS' if durations else 'warmup'
                        cv2.putText(picture, f'{backend.upper()} | {rate} | Q: cancel', (8,18),
                                    cv2.FONT_HERSHEY_SIMPLEX, .45, (0,255,160), 1, cv2.LINE_AA)
                        if not show_local_frame(picture):
                            cancelled = True
                            break
                        if i >= 10:
                            durations.append(time.perf_counter()-start)
                            hand_modes.add(hand['mode'])
            except KeyboardInterrupt:
                cancelled = True
            finally:
                stop_webcam()
                if durations:
                    report['live'][backend] = dict(frames=len(durations), fps=len(durations)/sum(durations),
                                                   p95_ms=float(np.percentile(durations,95)*1000),
                                                   hand_modes=sorted(hand_modes), cancelled=cancelled)
                    print(backend, report['live'][backend])
                (run_dir / 'results.json').write_text(json.dumps(report, indent=2))
            if cancelled:
                break
        if all(name in report['live'] for name in ('pytorch', 'onnx')):
            a, b = report['live']['pytorch'], report['live']['onnx']
            if (a['frames'] == b['frames'] == live_frames and not a['cancelled'] and not b['cancelled']
                    and a['hand_modes'] == b['hand_modes']):
                print(f'Live FPS change: {(b["fps"]/a["fps"]-1)*100:+.1f}%. '
                      'Separate camera runs; repeat in reverse order to check thermal/scene effects.')
            else:
                print('Live comparison incomplete or hand modes differed; do not claim an FPS improvement.')
    print(f'Saved experiment only: {run_dir / "results.json"}')
    return report


onnx_experiment_results = run_onnx_experiment(
    live=ONNX_LIVE_COMPARE, image_path=ONNX_IMAGE_PATH,
    count=ONNX_BENCHMARK_FRAMES, live_frames=ONNX_LIVE_FRAMES,
)
