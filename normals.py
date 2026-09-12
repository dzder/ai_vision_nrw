"""Image-space normals: x right, y down, z positive; no camera calibration."""
import cv2
import numpy as np


def compute_normals(depth, strength=40.0):
    """HxW relative depth -> HxWx3 float32 unit XYZ normals.

    Derivatives are per pixel (3x3 Sobel / 8). Strength controls the
    otherwise unknown ratio between relative depth and image coordinates.
    """
    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim != 2 or min(depth.shape) < 3 or not np.isfinite(depth).all():
        raise ValueError("Depth must be a finite HxW array at least 3x3.")
    if not np.isfinite(strength) or strength < 0:
        raise ValueError("Strength must be finite and nonnegative.")
    smooth = cv2.GaussianBlur(depth, (5, 5), 0)
    dx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3, scale=1 / 8)
    dy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3, scale=1 / 8)
    normals = np.stack((-strength * dx, -strength * dy, np.ones_like(depth)), axis=-1)
    normals /= np.linalg.norm(normals, axis=-1, keepdims=True)
    return normals


def normals_to_rgb(normals):
    """XYZ [-1,1] -> RGB uint8. Convert to BGR only at OpenCV display."""
    return np.rint(np.clip((normals + 1) * 127.5, 0, 255)).astype(np.uint8)


def sample_normal(depth, normals, point, radius=2):
    """Summarize a small patch; renormalize its mean direction to unit length.

    Relative depth is a per-frame visualization proxy, never metres.
    """
    h, w = depth.shape
    x, y = point
    if not (0 <= x < w and 0 <= y < h):
        raise ValueError("Selected point is outside the depth map.")
    x, y = min(int(round(x)), w-1), min(int(round(y)), h-1)
    patch = np.s_[max(0, y-radius):min(h, y+radius+1),
                  max(0, x-radius):min(w, x+radius+1)]
    normal = normals[patch].mean(axis=(0, 1))
    normal /= max(float(np.linalg.norm(normal)), 1e-8)
    return normal, float(np.median(depth[patch]))
