"""Depth Anything V2 Small inference: BGR frame -> aligned float32 relative depth proxy.

Values in [0,1] increase away from the camera; these are NOT metres.
Raw model output is affine-ambiguous inverse depth, so reversing its
normalized ordering is a visualization proxy, not metric reconstruction.
"""
import cv2
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"


class DepthEstimator:
    def __init__(self, device="auto", input_size=518):
        if input_size < 140 or input_size % 14:
            raise ValueError("Input size must be a multiple of 14 and at least 140.")
        self.input_size = input_size
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable. Install CUDA PyTorch or use --device cpu.")
        self.device = torch.device(device)
        # CUDA fp16 reduces memory use; actual speed/quality depend on hardware.
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.processor = AutoImageProcessor.from_pretrained(MODEL_ID)
        self.processor.size = {"height": input_size, "width": input_size}
        self.model_shape = None
        self.model = (
            AutoModelForDepthEstimation.from_pretrained(MODEL_ID, torch_dtype=self.dtype)
            .to(self.device)
            .eval()
        )

    @torch.inference_mode()
    def infer(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # No PIL round trip: the image processor accepts a numpy array
        # directly, and we keep everything as tensors on `self.device`
        # until the very last step.
        inputs = self.processor(images=rgb, return_tensors="pt")
        self.model_shape = tuple(inputs["pixel_values"].shape[-2:])
        inputs = {k: v.to(self.device, self.dtype) for k, v in inputs.items()}
        prediction = self.model(**inputs).predicted_depth  # (1, h', w') at model resolution

        # Upsample on-device with torch instead of the two CPU-side PIL
        # resizes in the pipeline version (one inside its postprocess,
        # one redundant one after).
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1).float(), size=(h, w), mode="bilinear", align_corners=False
        ).squeeze()
        inverse = prediction.detach().cpu().numpy()

        if not np.isfinite(inverse).all():
            raise RuntimeError("Depth model returned non-finite values.")
        lo, hi = np.percentile(inverse, [2, 98])
        if hi - lo < 1e-6:
            return np.zeros_like(inverse, dtype=np.float32)
        return (1.0 - np.clip((inverse - lo) / (hi - lo), 0, 1)).astype(np.float32)
