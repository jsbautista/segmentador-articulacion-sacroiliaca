"""Visualización de cortes: ventana de intensidad, superposición y colores por clase.

Funciones puras de NumPy/Pillow, sin Streamlit. La imagen y la máscara pasan por
exactamente las mismas funciones de selección y orientación, de modo que la
correspondencia espacial se conserva.
"""
from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
from PIL import Image

# Colores fijos por etiqueta: no cambian de un corte a otro (el prototipo los
# reescalaba por corte con un colormap continuo).
PALETTE = [
    (240, 112, 48), (64, 196, 255), (112, 214, 104), (250, 214, 64),
    (190, 120, 255), (255, 120, 190), (0, 170, 150), (200, 200, 200),
]


def class_color(label: int) -> tuple[int, int, int]:
    return PALETTE[(int(label) - 1) % len(PALETTE)]


def hex_color(label: int) -> str:
    r, g, b = class_color(label)
    return f"#{r:02x}{g:02x}{b:02x}"


def take_slice(arr: np.ndarray, index: int, axis: int) -> np.ndarray:
    return np.take(arr, index, axis=axis)


def orient(sl: np.ndarray, display_cfg: dict) -> np.ndarray:
    """Orientación de pantalla. Se aplica igual a imagen y máscara."""
    if display_cfg.get("transpose", True):
        sl = sl.T
    if display_cfg.get("flip_vertical", True):
        sl = sl[::-1]
    return np.ascontiguousarray(sl)


def pixel_aspect(spacing: tuple[float, float, float], axis: int, display_cfg: dict) -> float:
    """Alto/ancho físico de un píxel mostrado (1.0 = cuadrado)."""
    a0, a1 = [a for a in range(3) if a != axis]
    if display_cfg.get("transpose", True):
        horizontal, vertical = spacing[a0], spacing[a1]
    else:
        horizontal, vertical = spacing[a1], spacing[a0]
    return float(vertical / horizontal)


def window_to_uint8(sl: np.ndarray, display_range: tuple[float, float],
                    brightness: float = 0.0, contrast: float = 1.0) -> np.ndarray:
    """Ventana lineal. La misma `display_range` (p1-p99 del volumen) para todos los cortes."""
    lo, hi = display_range
    width = max(hi - lo, 1e-6)
    center = (lo + hi) / 2.0 - brightness * width
    half = max(width / 2.0 / max(contrast, 1e-3), 1e-6)
    out = (sl.astype(np.float32) - (center - half)) / (2.0 * half)
    return (np.clip(out, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


@dataclass(frozen=True)
class ViewParams:
    display_range: tuple[float, float]
    brightness: float = 0.0
    contrast: float = 1.0
    show_mask: bool = True
    mask_only: bool = False
    opacity: float = 0.45
    outline: bool = False
    visible_labels: tuple[int, ...] = ()


def _edge(m: np.ndarray) -> np.ndarray:
    p = np.pad(m, 1, constant_values=False)
    inner = p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:] & m
    return m & ~inner


def _blend(rgb: np.ndarray, mask: np.ndarray, p: ViewParams) -> np.ndarray:
    present = [int(l) for l in np.unique(mask) if l > 0 and int(l) in p.visible_labels]
    if not present:
        return rgb
    out = rgb.astype(np.float32)
    alpha = max(p.opacity, 0.6) if p.outline else p.opacity
    for lab in present:
        sel = mask == lab
        if p.outline:
            sel = _edge(sel)
        color = np.array(class_color(lab), dtype=np.float32)
        out[sel] = out[sel] * (1.0 - alpha) + color * alpha
    return (out + 0.5).astype(np.uint8)


def _fix_aspect(rgb: np.ndarray, aspect: float) -> np.ndarray:
    if abs(aspect - 1.0) < 0.02:
        return rgb
    h, w = rgb.shape[:2]
    return np.asarray(Image.fromarray(rgb).resize((w, max(1, round(h * aspect))), Image.BILINEAR))


def render(image2d: np.ndarray, mask2d: np.ndarray | None, p: ViewParams, aspect: float = 1.0) -> np.ndarray:
    """Compone el corte como RGB uint8. `image2d` y `mask2d` ya vienen orientados."""
    use_mask = mask2d is not None and p.show_mask
    if use_mask and p.mask_only:
        rgb = np.zeros(image2d.shape + (3,), dtype=np.uint8)
    else:
        gray = window_to_uint8(image2d, p.display_range, p.brightness, p.contrast)
        rgb = np.repeat(gray[..., None], 3, axis=2)
    if use_mask:
        rgb = _blend(rgb, mask2d, p)
    return _fix_aspect(rgb, aspect)


def to_png(rgb: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return buf.getvalue()
