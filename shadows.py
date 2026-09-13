"""Level 04 Dynamic Shadows — Screen-Space Ray Marching.

SmoothLight  — physics-based position smoother (inertia + friction)
ShadowEngine — ray-march occlusion with Bayer dithering, thickness test,
               quadratic contact hardening, bilateral blur, and asymmetric
               temporal filtering (fast appearance, slow disappearance)
apply_shadows — composite shadows onto a relit image with bloom + vignette
"""
import cv2
import numpy as np


class SmoothLight:
    """Smooth a raw hand-tracker position with spring-damper dynamics.

    Absorbs the ±5 px jitter inherent to MediaPipe while remaining
    responsive to intentional hand movement.  When the hand disappears,
    the position decelerates with friction instead of snapping.
    """

    def __init__(self, w, h):
        self.w, self.h = w, h
        self.pos = np.array([w / 2.0, h / 3.0, -0.75], dtype=np.float64)
        self.vel = np.zeros(3, dtype=np.float64)
        self.initialized = False
        self.lost_frames = 0

    def update(self, raw_pos=None):
        """Feed a raw (x, y, z) each frame; returns the smoothed tuple."""
        if raw_pos is not None:
            target = np.array(raw_pos, dtype=np.float64)
            self.lost_frames = 0
            if not self.initialized:
                self.pos = target.copy()
                self.initialized = True
                return tuple(self.pos)
            diff = target - self.pos
            dist = np.linalg.norm(diff[:2])
            # Large motion → responsive; small motion → absorbs jitter.
            alpha = np.clip(0.06 + dist * 0.004, 0.06, 0.40)
            self.vel = alpha * diff + (1.0 - alpha) * self.vel * 0.75
            self.pos += self.vel
        else:
            self.lost_frames += 1
            friction = max(0.88 - self.lost_frames * 0.002, 0.0)
            self.vel *= friction
            self.pos += self.vel
            if self.lost_frames > 50:
                neutral = np.array([self.w / 2.0, self.h / 3.0, -0.75])
                self.pos += (neutral - self.pos) * 0.002

        self.pos[0] = np.clip(self.pos[0], 0, self.w)
        self.pos[1] = np.clip(self.pos[1], 0, self.h)
        self.pos[2] = np.clip(self.pos[2], -3.0, -0.05)
        return tuple(self.pos)


# Pre-built Bayer 4×4 dithering matrix.
_BAYER4 = np.array([
    [ 0,  8,  2, 10],
    [12,  4, 14,  6],
    [ 3, 11,  1,  9],
    [15,  7, 13,  5],
], dtype=np.float32) / 16.0


class ShadowEngine:
    """Screen-space ray-march shadow map.

    For every pixel, march toward the light in *num_steps* increments.
    If a scene sample closer to the camera is found within a thin
    *thickness* band, the pixel is considered occluded.

    Temporal stability comes from asymmetric EMA: shadow darkening is
    fast (*temporal_rise*) while brightening is slow (*temporal_fall*),
    so shadows never pop or vanish abruptly.
    """

    def __init__(self, *, num_steps=10, shadow_bias=0.008, thickness=0.10,
                 shadow_intensity=0.88, downscale=4, blur_d=7,
                 temporal_rise=0.50, temporal_fall=0.10):
        if num_steps < 1:
            raise ValueError("num_steps must be >= 1.")
        self.num_steps = num_steps
        self.shadow_bias = shadow_bias
        self.thickness = thickness
        self.shadow_intensity = shadow_intensity
        self.downscale = max(1, int(downscale))
        self.blur_d = blur_d
        self.temporal_rise = temporal_rise
        self.temporal_fall = temporal_fall
        self._prev = None

    def compute(self, depth, light_xy, light_z):
        """Return a float32 (H, W) mask: 0 = full shadow, 1 = full light."""
        depth = np.asarray(depth, dtype=np.float32)
        H, W = depth.shape
        lx, ly = float(light_xy[0]), float(light_xy[1])
        lz = float(light_z)
        ds = self.downscale

        # ── down-sample for performance ──
        if ds > 1:
            Ws, Hs = W // ds, H // ds
            depth_s = cv2.resize(depth, (Ws, Hs), interpolation=cv2.INTER_AREA)
            lx_s, ly_s = lx / ds, ly / ds
        else:
            depth_s = depth
            Hs, Ws = H, W
            lx_s, ly_s = lx, ly

        Y, X = np.mgrid[:Hs, :Ws].astype(np.float32)

        dir_x = lx_s - X
        dir_y = ly_s - Y
        dir_z = lz - depth_s

        inv = 1.0 / self.num_steps
        sx = dir_x * inv
        sy = dir_y * inv
        sz = dir_z * inv

        # Tile the Bayer matrix
        jitter = np.tile(_BAYER4, (Hs // 4 + 1, Ws // 4 + 1))[:Hs, :Ws]

        occlusion = np.zeros((Hs, Ws), dtype=np.float32)
        for i in range(1, self.num_steps + 1):
            t = i + jitter
            cx = np.clip((X + sx * t).astype(np.int32), 0, Ws - 1)
            cy = np.clip((Y + sy * t).astype(np.int32), 0, Hs - 1)
            ray_z = depth_s + sz * t
            scene_z = depth_s[cy, cx]
            diff = scene_z - ray_z
            blocked = (diff > self.shadow_bias) & (diff < self.thickness)
            hardness = (1.0 - i * inv) ** 1.8
            occlusion = np.maximum(occlusion,
                                   blocked.astype(np.float32) * hardness)

        shadow = 1.0 - np.clip(occlusion, 0, 1) * self.shadow_intensity

        # ── up-sample ──
        if ds > 1:
            shadow = cv2.resize(shadow, (W, H), interpolation=cv2.INTER_LINEAR)

        # Fast Gaussian blur instead of slow Bilateral filter
        shadow = cv2.GaussianBlur(shadow, (self.blur_d | 1, self.blur_d | 1), 0)
        
        # Gentle gamma for natural contrast.
        shadow = np.power(np.clip(shadow, 1e-6, 1.0), 0.88)

        # ── asymmetric temporal filter ──
        if self._prev is not None and self._prev.shape == shadow.shape:
            alpha = np.where(shadow < self._prev,
                             self.temporal_rise, self.temporal_fall)
            shadow = alpha * shadow + (1.0 - alpha) * self._prev
        self._prev = shadow.copy()

        return np.clip(shadow, 0, 1).astype(np.float32)


# ── compositing helpers ──────────────────────────────────────────────

def apply_shadows(relit_bgr, frame_bgr, shadow, *, ambient=0.15,
                  bloom_strength=0.6):
    """Composite *shadow* onto a relit image with fast bloom and vignette."""
    relit = relit_bgr.astype(np.float32) / 255.0
    base = frame_bgr.astype(np.float32) / 255.0
    ambient_layer = base * ambient
    direct = np.clip(relit - ambient_layer, 0, 1)
    lit = ambient_layer + direct * shadow[..., np.newaxis]

    # Fast Bloom: downscale by 4, isolate bright, blur, upscale, blend.
    H, W = lit.shape[:2]
    lit_small = cv2.resize(lit, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
    luma = 0.299 * lit_small[..., 0] + 0.587 * lit_small[..., 1] + 0.114 * lit_small[..., 2]
    bright = np.clip((luma - 0.50) / 0.50, 0, 1)
    bloom_small = cv2.GaussianBlur(lit_small * bright[..., np.newaxis], (15, 15), sigmaX=0)
    bloom = cv2.resize(bloom_small, (W, H), interpolation=cv2.INTER_LINEAR)
    lit = lit + bloom * bloom_strength

    # Fast Vignette: compute 1D arrays and broadcast.
    cy, cx = H / 2.0, W / 2.0
    y = (np.arange(H, dtype=np.float32) - cy) / cy
    x = (np.arange(W, dtype=np.float32) - cx) / cx
    # Broadcasting: (H, 1) + (1, W) -> (H, W)
    r = np.sqrt(y[:, np.newaxis]**2 + x[np.newaxis, :]**2)
    vignette = 1.0 - np.clip(r - 0.7, 0, 1) * 0.45
    lit = lit * vignette[..., np.newaxis]

    return np.clip(lit * 255, 0, 255).astype(np.uint8)
