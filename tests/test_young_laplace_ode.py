import numpy as np
import pytest
from menipy.math.young_laplace import young_laplace_ode

def test_young_laplace_sphere_limit():
    """Test that with beta=0, the shape approaches a circle radius R0."""
    R0 = 5.0
    beta = 0.0
    
    # Generate profile
    profile = young_laplace_ode(np.array([R0, beta]), physics={})
    
    # Expected points should fall on x^2 + (y-R0)^2 = R0^2
    # So r^2 + (z-R0)^2 = R0^2
    # The ODE starts at r=0, z=0 (apex)
    
    r = profile[:, 0]
    z = profile[:, 1]
    
    distances_to_center = np.sqrt(r**2 + (z - R0)**2)
    
    # At beta=0, it should be a perfect circle
    np.testing.assert_allclose(distances_to_center, R0, rtol=1e-3, atol=1e-3)

def test_young_laplace_positive_beta():
    """Test standard pendant drop with positive beta."""
    R0 = 2.0
    beta = 0.3
    
    profile = young_laplace_ode(np.array([R0, beta]), physics={})
    
    assert profile.shape[0] > 0
    assert profile.shape[1] == 2
    
    # The profile should be symmetric
    n = len(profile)
    mid = n // 2
    r_left = profile[:mid, 0]
    r_right = profile[-mid:, 0]
    
    # Check bounds
    assert np.all(r_left < 1e-5) # Left side is negative or near 0
    assert np.all(r_right > -1e-5) # Right side is positive or near 0
    
    # Apex is approx at mid
    assert abs(profile[mid, 0]) < 1e-2
    assert abs(profile[mid, 1]) < 1e-2

def test_young_laplace_single_param_fallback():
    """Test fallback when only R0 is provided."""
    R0 = 5.0
    profile = young_laplace_ode(np.array([R0]), physics={})
    
    r = profile[:, 0]
    z = profile[:, 1]
    distances = np.sqrt(r**2 + (z - R0)**2)
    np.testing.assert_allclose(distances, R0, rtol=1e-3, atol=1e-3)
