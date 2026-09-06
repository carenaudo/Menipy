import math

import numpy as np

from menipy.common.geometry import detect_baseline_reflection_cusp
from menipy.math.young_laplace import sessile_young_laplace_ode


def test_detect_baseline_reflection_cusp():
    # Synthetic droplet + mirror reflection below baseline y = 300
    # Top droplet: y from 100 to 300, width narrows to 80 px at y=300
    # Bottom reflection: y from 300 to 450, width widens to 110 px
    y_drop = np.linspace(100, 300, 50)
    w_drop = 40.0 + 30.0 * np.sin(np.pi * (y_drop - 100) / 200.0)  # maximum at mid, narrows to 40 at y=300
    # Add a definite pinch at y=300 (width = 40)
    # Reflection below y=300:
    y_refl = np.linspace(302, 420, 30)
    w_refl = 40.0 + 25.0 * np.sin(np.pi * (y_refl - 302) / 118.0)

    y_all = np.concatenate([y_drop, y_refl])
    w_all = np.concatenate([w_drop, w_refl])

    x_center = 250.0
    x_left = x_center - w_all
    x_right = x_center + w_all

    left_branch = np.column_stack([x_left, y_all])
    right_branch = np.column_stack([x_right[::-1], y_all[::-1]])
    contour = np.vstack([left_branch, right_branch])

    result = detect_baseline_reflection_cusp(contour, window_y_px=3.0)
    assert result is not None
    p1, p2, conf = result
    assert abs(p1[1] - 300.0) < 10.0
    assert abs(p2[1] - 300.0) < 10.0
    assert conf > 0.5


def test_detect_baseline_reflection_cusp_negative():
    # Regular sessile drop without reflection (flat bottom or truncation)
    theta = np.linspace(0, np.pi, 60)
    xs = 200.0 + 80.0 * np.cos(theta)
    ys = 250.0 - 60.0 * np.sin(theta)
    contour = np.column_stack([xs, ys])

    # No reflection below baseline
    result = detect_baseline_reflection_cusp(contour, window_y_px=3.0)
    assert result is None


def test_sessile_young_laplace_ode():
    R0_mm = 2.5
    Bo = 0.25
    target_height_mm = 1.8

    profile = sessile_young_laplace_ode(
        params=np.array([R0_mm, Bo]),
        physics={"rho1": 1000.0, "rho2": 1.2, "g": 9.80665},
        geometry={"height_mm": target_height_mm},
    )

    assert profile.ndim == 2
    assert profile.shape[0] > 10
    assert abs(np.min(profile[:, 1])) < 1e-4  # apex at z=0
    # Reaches near target height
    assert abs(np.max(profile[:, 1]) - target_height_mm) < 0.2
    # Symmetric around r=0
    assert math.isclose(float(np.min(profile[:, 0])), -float(np.max(profile[:, 0])), rel_tol=1e-2)
