"""Drop analysis module."""

from .commons import compute_drop_metrics, find_apex_index, extract_external_contour
from .plotting import save_contour_sides_image, save_contour_side_profiles
from .pendant import compute_metrics as compute_pendant_metrics
from .sessile import compute_metrics as compute_sessile_metrics, smooth_contour_segment
from .sessile_alt import compute_metrics as compute_sessile_metrics_alt
from ..detection.needle import detect_vertical_edges

__all__ = [
    "compute_drop_metrics",
    "compute_pendant_metrics",
    "compute_sessile_metrics",
    "compute_sessile_metrics_alt",
    "find_apex_index",
    "extract_external_contour",
    "detect_vertical_edges",
    "save_contour_sides_image",
    "save_contour_side_profiles",
    "smooth_contour_segment",
]
