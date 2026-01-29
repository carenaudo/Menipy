# src/adsa/common/overlay.py
"""
Overlay drawing utilities for visualizing analysis results on images.
"""
from __future__ import annotations
from typing import Iterable, Tuple, Union, Dict, Any
import numpy as np

try:
    import cv2  # type: ignore
except Exception as e:  # pragma: no cover
    cv2 = None
    _IMPORT_ERROR = e

Color = Union[Tuple[int, int, int], str]  # BGR or name

_COLOR_MAP = {
    "red": (0, 0, 255),
    "green": (0, 255, 0),
    "blue": (255, 0, 0),
    "cyan": (255, 255, 0),
    "yellow": (0, 255, 255),
    "magenta": (255, 0, 255),
    "white": (255, 255, 255),
    "black": (0, 0, 0),
    "orange": (0, 165, 255),
}


def _bgr(c: Color) -> Tuple[int, int, int]:
    if isinstance(c, str):
        return _COLOR_MAP.get(c.lower(), (255, 255, 255))
    return tuple(int(v) for v in c)  # type: ignore


def _ensure_bgr(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return np.dstack([img, img, img])  # gray -> BGR
    if img.ndim == 3 and img.shape[2] == 3:
        return img.copy()
    raise ValueError("overlay: expected (H,W) or (H,W,3) image")


def _require_cv2():
    if cv2 is None:
        raise RuntimeError(
            f"OpenCV (cv2) is required for overlay drawing but not available: {_IMPORT_ERROR}"
        )


def _as_ndarray(img_like: Any) -> np.ndarray:
    """
    Accept either a raw numpy image or a Frame-like object with an `.image` attribute.
    """
    try:
        if hasattr(img_like, "image"):
            return img_like.image  # type: ignore[return-value]
    except Exception:
        pass
    return img_like  # type: ignore[return-value]


# ---- commands ---------------------------------------------------------------
# Each command is a dict with a "type" field and the needed params.
# Types: line, polyline, cross, text, circle, scatter


def draw_overlay(
    base_img: np.ndarray,
    commands: Iterable[Dict[str, Any]],
    alpha: float = 0.6,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (overlay_only, composited_image).
    - overlay_only: BGR image with drawings on black
    - composited_image: (1-alpha)*base + alpha*overlay
    """
    _require_cv2()
    img = _ensure_bgr(base_img)
    # Ensure we have a 3-channel image for the overlay
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("overlay: expected 3-channel BGR image")
    H, W = img.shape[:2]
    # Ensure overlay has same shape as the 3-channel BGR image
    overlay: np.ndarray = np.zeros((H, W, 3), dtype=img.dtype)  # type: ignore[assignment]

    for cmd in commands:
        typ = cmd.get("type")
        if typ == "line":
            p1 = tuple(map(int, cmd["p1"]))
            p2 = tuple(map(int, cmd["p2"]))
            color = _bgr(cmd.get("color", "cyan"))
            th = int(cmd.get("thickness", 2))
            cv2.line(overlay, p1, p2, color, th, lineType=cv2.LINE_AA)

        elif typ == "polyline":
            pts_poly: np.ndarray = np.asarray(cmd["points"], dtype=np.int32).reshape(-1, 1, 2)
            color = _bgr(cmd.get("color", "yellow"))
            th = int(cmd.get("thickness", 2))
            closed = bool(cmd.get("closed", True))
            cv2.polylines(overlay, [pts_poly], closed, color, th, lineType=cv2.LINE_AA)

        elif typ == "cross":
            cx, cy = map(int, cmd["p"])
            size = int(cmd.get("size", 6))
            color = _bgr(cmd.get("color", "red"))
            th = int(cmd.get("thickness", 2))
            cv2.line(overlay, (cx - size, cy), (cx + size, cy), color, th, cv2.LINE_AA)
            cv2.line(overlay, (cx, cy - size), (cx, cy + size), color, th, cv2.LINE_AA)

        elif typ == "text":
            x, y = map(int, cmd["p"])
            s = str(cmd.get("text", ""))
            color = _bgr(cmd.get("color", "white"))
            scale = float(cmd.get("scale", 0.5))
            th = int(cmd.get("thickness", 1))
            cv2.putText(
                overlay,
                s,
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                scale,
                color,
                th,
                cv2.LINE_AA,
            )

        elif typ == "circle":
            c = tuple(map(int, cmd["center"]))
            r = int(cmd.get("radius", 10))
            color = _bgr(cmd.get("color", "magenta"))
            th = int(cmd.get("thickness", 2))
            cv2.circle(overlay, c, r, color, th, cv2.LINE_AA)

        elif typ == "scatter":
            pts_scatter: np.ndarray = np.asarray(cmd["points"], dtype=int).reshape(-1, 2)
            color = _bgr(cmd.get("color", "white"))
            r = int(cmd.get("radius", 2))
            th = int(cmd.get("thickness", -1))
            for x, y in pts_scatter:
                cv2.circle(overlay, (int(x), int(y)), r, color, th, cv2.LINE_AA)

        else:
            raise ValueError(f"overlay: unknown command type '{typ}'")

    # composite
    comp_raw = cv2.addWeighted(overlay, float(alpha), img, float(1.0 - alpha), 0.0)
    comp = _ensure_bgr(comp_raw)
    comp = np.asarray(comp)
    return overlay, comp


def run(ctx, *, commands: Iterable[Dict[str, Any]], alpha: float = 0.6):
    """
    Stage entrypoint: draw over ctx.frames[0] (or ctx.frames) and attach:
      - ctx.overlay (BGR)
      - ctx.preview (BGR composited)
    """
    frames = ctx.frames if isinstance(ctx.frames, list) else [ctx.frames]
    if not frames:
        raise ValueError("overlay.run: no frames available")
    base_img = _as_ndarray(frames[0])
    overlay, comp = draw_overlay(base_img, commands, alpha=alpha)
    ctx.overlay = overlay
    ctx.preview = comp
    return ctx
