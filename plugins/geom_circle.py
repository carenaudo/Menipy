from menipy.common.registry import register_geometry
import numpy as np


def geom_circle(ctx):
    # attach a simple geometry approximation if contour present
    contour = getattr(ctx, "contour", None)
    if contour is not None:
        pts = contour.xy if hasattr(contour, "xy") else contour
        # simple bounding-circle estimate
        cx, cy = float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1]))
        ctx.geometry = {"center": (cx, cy)}
    return ctx


register_geometry("circle_estimate", geom_circle)
