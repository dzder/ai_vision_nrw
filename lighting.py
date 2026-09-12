"""Point-light shading of the Level 1 orthographic relative-depth height field.

X right, Y down, Z away. World units are one longest image dimension.
This is an artistic geometry proxy, not a calibrated reconstruction.
"""
import cv2
import numpy as np


def surface_points(depth, strength=40.0):
    """Match compute_normals' smoothed surface and depth-to-pixel scale."""
    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim != 2 or min(depth.shape) < 3 or not np.isfinite(depth).all():
        raise ValueError("Depth must be finite HxW, at least 3x3.")
    if not np.isfinite(strength) or strength < 0:
        raise ValueError("Strength must be finite and nonnegative.")
    h, w = depth.shape
    y, x = np.mgrid[:h, :w].astype(np.float32)
    scale = float(max(h, w))
    z = cv2.GaussianBlur(depth, (5, 5), 0) * (strength / scale)
    return np.stack(((x-(w-1)/2)/scale, (y-(h-1)/2)/scale, z), axis=-1)


def _unit(vectors):
    return vectors / np.maximum(np.linalg.norm(vectors, axis=-1, keepdims=True), 1e-8)


def shade(frame_bgr, points, normals, light_position, *, ambient=0.18,
          intensity=1.8, specular=0.35, shininess=48.0):
    """Return uint8 BGR: linear-light Lambert + Blinn-Phong, then sRGB.

Input normals are Level 1's +Z-oriented normals. Negating them yields the
camera-facing surface. Camera RGB approximates albedo and retains real lighting.
Softened inverse-square falloff avoids a singularity at the light position.
"""
    frame_bgr = np.asarray(frame_bgr)
    points = np.asarray(points, dtype=np.float32)
    normals = np.asarray(normals, dtype=np.float32)
    light = np.asarray(light_position, dtype=np.float32)
    parameters = np.asarray([ambient, intensity, specular, shininess])
    if (frame_bgr.ndim != 3 or frame_bgr.shape[-1] != 3 or frame_bgr.dtype != np.uint8
            or points.shape != frame_bgr.shape or normals.shape != points.shape):
        raise ValueError("Frame must be uint8 HxWx3 BGR; points/normals must match.")
    if (light.shape != (3,) or not np.isfinite(light).all()
            or not np.isfinite(points).all() or not np.isfinite(normals).all()
            or not np.isfinite(parameters).all() or (parameters < 0).any()
            or shininess < 1):
        raise ValueError("Geometry/light must be finite; gains >= 0 and shininess >= 1.")
    n = -_unit(normals)
    to_light = light - points
    distance_squared = np.sum(to_light * to_light, axis=-1)
    l = _unit(to_light)
    # Orthographic geometry uses parallel rays toward the viewer (-Z).
    view = np.array([0, 0, -1], dtype=np.float32)
    half_vector = _unit(l + view)
    diffuse = np.maximum(np.sum(n * l, axis=-1), 0)
    highlight = np.maximum(np.sum(n * half_vector, axis=-1), 0) ** shininess
    highlight *= (diffuse > 0) & (n[..., 2] < 0)
    attenuation = intensity / (1.0 + distance_squared)
    srgb = frame_bgr.astype(np.float32) / 255.0
    albedo = np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)
    linear = albedo * (ambient + attenuation * diffuse)[..., None]
    linear += (attenuation * specular * highlight)[..., None]
    linear = np.clip(linear, 0, 1)
    output = np.where(linear <= 0.0031308, 12.92 * linear,
                      1.055 * linear ** (1 / 2.4) - 0.055)
    return np.rint(np.clip(output * 255, 0, 255)).astype(np.uint8)


class LightController:
    """Mouse over panel three controls XY; wheel or brackets adjust Z."""
    def __init__(self):
        self.shape = None
        self.reset()

    def reset(self):
        self.position = np.array([-0.35, -0.25, -0.75], dtype=np.float32)
        self.intensity = 1.8
        self.specular = 0.35

    def move_z(self, delta):
        self.position[2] = np.clip(self.position[2] + delta, -3.0, -0.05)

    def on_mouse(self, event, x, y, flags, userdata):
        if self.shape is None:
            return
        h, w = self.shape
        if not (2*w <= x < 3*w and 0 <= y < h):
            return
        if event in (cv2.EVENT_MOUSEMOVE, cv2.EVENT_LBUTTONDOWN):
            scale = float(max(h, w))
            self.position[:2] = [(x-2*w-(w-1)/2)/scale, (y-(h-1)/2)/scale]
        elif event == cv2.EVENT_MOUSEWHEEL:
            delta = (flags >> 16) & 0xffff
            if delta >= 0x8000:
                delta -= 0x10000
            self.move_z(0.1 * delta / 120)

    def on_key(self, key):
        if key == ord("["):
            self.move_z(-0.1)
        elif key == ord("]"):
            self.move_z(0.1)
        elif key in (ord("+"), ord("=")):
            self.intensity = min(5.0, self.intensity + 0.1)
        elif key == ord("-"):
            self.intensity = max(0.0, self.intensity - 0.1)
        elif key in (ord("s"), ord("S")):
            self.specular = 0.0 if self.specular else 0.35
        elif key in (ord("r"), ord("R")):
            self.reset()

    def draw(self, display):
        h, w = self.shape
        scale = max(h, w)
        x = int(round(self.position[0]*scale + (w-1)/2))
        y = int(round(self.position[1]*scale + (h-1)/2))
        # Clip the marker to its panel (including extreme aspect ratios).
        cv2.drawMarker(display[:h, 2*w:3*w], (x, y), (0, 220, 255),
                       cv2.MARKER_CROSS, 16, 1, cv2.LINE_AA)
