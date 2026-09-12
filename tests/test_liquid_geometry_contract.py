"""Regression tests for the authoritative liquid-side geometry contract."""

import numpy as np

from menipy.common.liquid_boundary import build_straight_liquid_geometry


def _cap(*, tilted=False):
    theta = np.linspace(np.pi, 0, 101)
    surface = np.column_stack([50 + 40 * np.cos(theta), 59.5 - 30 * np.sin(theta)])
    solid_return = np.column_stack([np.linspace(90, 10, 41), np.full(41, 61.0)])
    contour = np.vstack([surface, solid_return])
    line = ((0.0, 60.0), (100.0, 60.0))
    if tilted:
        line = ((0.0, 55.0), (100.0, 65.0))
        # Translate all points to retain a clear apex-side cap for the tilted line.
        contour[:, 1] += 0.1 * (contour[:, 0] - 50)
    return contour, line


def test_complete_geometry_keeps_only_the_apex_side_and_separate_closure():
    contour, line = _cap()
    result = build_straight_liquid_geometry(contour, line, apex=(50.0, 30.0))

    assert result.status == "complete"
    assert result.contact_points is not None
    assert result.observed_surface is not None
    assert result.closed_region is not None
    assert np.max(result.observed_surface[:, 1]) < 60.0
    # The closing segment is not an observed fitting sample.
    assert len(result.closed_region) == len(result.observed_surface) + 3


def test_tilted_line_uses_signed_distance_not_average_y():
    contour, line = _cap(tilted=True)
    result = build_straight_liquid_geometry(contour, line, apex=(50.0, 30.0))

    assert result.status == "complete"
    p1, p2 = map(np.asarray, line)
    direction = p2 - p1
    signed = direction[0] * (result.observed_surface[:, 1] - p1[1]) - direction[1] * (result.observed_surface[:, 0] - p1[0])
    assert np.all(signed < 1e-6)


def test_ambiguous_crossings_are_unresolved_not_automatically_closed():
    contour, line = _cap()
    # A second excursion through the substrate creates four crossings.
    contour = np.vstack([contour[:60], [[50, 70], [50, 50]], contour[60:]])
    result = build_straight_liquid_geometry(contour, line, apex=(50.0, 30.0))

    assert result.status == "unresolved"
    assert "contact_boundary_ambiguous_crossings" in result.rejection_reasons
