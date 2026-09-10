"""Exact adaptive profiles and events after direct state indexing."""

import numpy as np
import pytest

from menipy.math import young_laplace as yl
from menipy.pipelines.pendant import strict_young_laplace as strict
from tests import ode_state_reference as reference


@pytest.mark.parametrize("name", ["young_laplace_ode", "sessile_young_laplace_ode"])
@pytest.mark.parametrize("params", [[2, 0.25], [1, 0.8], [2, -0.1], [0, 0.3]])
@pytest.mark.parametrize("geometry", [None, {"height_mm": 1.8}])
def test_generic_profiles(name, params, geometry):
    np.testing.assert_array_equal(getattr(yl, name)(params, {}, geometry),
                                  getattr(reference, name)(params, {}, geometry))


@pytest.mark.parametrize("branch", ["full", "right"])
@pytest.mark.parametrize("kwargs", [{}, {"target_height_mm": 1.8},
                                    {"needle_radius_mm": 0.3},
                                    {"target_height_mm": 1.8, "needle_radius_mm": 0.3}])
def test_strict_profiles_and_events(branch, kwargs):
    actual, meta = strict.integrate_young_laplace_profile_mm(
        2, 0.25, branch=branch, return_metadata=True, **kwargs)
    expected, expected_meta = reference.integrate_young_laplace_profile_mm(
        2, 0.25, branch=branch, return_metadata=True, **kwargs)
    np.testing.assert_array_equal(actual, expected)
    assert meta == expected_meta
