"""Test Sessile Geometry.

Test module."""


import numpy as np
import pytest
from menipy.pipelines.sessile.geometry import clip_contour_to_substrate, _segment_intersection

"""
Tests for sessile drop geometry utilities.
Includes tests for segment intersection and contour clipping logic.
"""

def test_segment_intersection():
    """segment intersection.
    """
    # Crossing lines
    p1 = np.array([0, 0])
    p2 = np.array([2, 2])
    q1 = np.array([0, 2])
    q2 = np.array([2, 0])
    inter = _segment_intersection(p1, p2, q1, q2)
    np.testing.assert_allclose(inter, [1, 1])

    # Parallel lines
    p1 = np.array([0, 0])
    p2 = np.array([1, 0])
    q1 = np.array([0, 1])
    q2 = np.array([1, 1])
    inter = _segment_intersection(p1, p2, q1, q2)
    assert inter is None

    # Non-crossing segments (but line intersection exists)
    p1 = np.array([0, 0])
    p2 = np.array([1, 1])
    q1 = np.array([2, 0])
    q2 = np.array([2, 1])
    # q is vertical line x=2. p line is y=x. intersection at (2,2).
    # But function checks if intersection is within p1-p2 segment?
    # Wait, implementation: t = qp_x_s / r_x_s. if 0<=t<=1 returns point.
    # So it checks if intersection is within FIRST segment (p1-p2).
    # Does it check second segment? Implementation:
    # Line q = q1 + u * u_vec. It doesn't check u.
    # So it intersects LINE q with SEGMENT p.
    
    inter = _segment_intersection(p1, p2, q1, q2)
    assert inter is None # (2,2) is outside p1-p2

def test_clip_contour_simple():
    """clip contour simple.
    """
    # Square contour crossing y=0
    # Points: (-1, 2), (-1, -1), (1, -1), (1, 2)
    # Substrate line: y=0
    # Apex: (0, 2)
    
    contour = np.array([
        [-1, 2],
        [-1, -1], # Crosses here between 0 and 1
        [1, -1],
        [1, 2]    # Crosses here between 2 and 3
    ], dtype=float)
    
    substrate = ((-2.0, 0.0), (2.0, 0.0))
    apex = (0.0, 2.0)
    
    # Expected: 
    # Starts at apex index (closest to 0,2 is index 0 or 3? Distances:
    # 0: (-1,2)->(0,2) d=1
    # 3: (1,2)->(0,2) d=1
    # Let's say apex_idx=0.
    
    # Walk right (forward): 0->1.
    # 0=(-1,2), 1=(-1,-1). Crosses y=0 at (-1,0).
    # So right_contact should be (-1,0). Arc ends.
    
    # Walk left (backward): 0->3.
    # 0=(-1,2), 3=(1,2). No cross.
    # 3->2. 3=(1,2), 2=(1,-1). Crosses y=0 at (1,0).
    # So left_contact should be (1,0). Arc ends.
    
    # New contour: left_contact, points in arc, right_contact
    # Arc is ... 3, 0 ...
    # So [ (1,0), (1,2), (-1,2), (-1,0) ]
    
    refined, contacts = clip_contour_to_substrate(contour, substrate, apex)
    
    assert contacts is not None
    # Sorting: left-to-right
    left, right = contacts
    np.testing.assert_allclose(left, [-1, 0], atol=1e-5)
    np.testing.assert_allclose(right, [1, 0], atol=1e-5)
    
    # Check refined contour points
    # Logic might produce them in specific order depending on start/end
    # With apex at (-1,2) (idx 0), filtered points should be above y=0
    assert np.all(refined[:, 1] >= -1e-5)

def test_clip_contour_tilted():
    """clip contour tilted.
    """
    # Tilted line y = x
    # Apex at (0, 1) (above line)
    # Contour is circle-ish
    # Points: (0, 1), (2, 1), (1, -1)
    
    # Line: (-1, -1) to (3, 3)
    substrate = ((-1.0, -1.0), (3.0, 3.0))
    apex = (0.0, 1.0)
    
    contour = np.array([
        [0, 1],   # Above y=x (0 < 1)
        [2, 1],   # Below y=x (2 > 1) -> Crossing 1
        [1, -1],  # Below
        [-1, 1],  # Above ( -1 < 1) -> Crossing 2
    ], dtype=float)
    
    # Apex index = 0
    # Forward 0->1: (0,1) to (2,1). Intersects y=x.
    # x = y. Line segment p=(0,1)+t*(2,0) -> (2t, 1).
    # 2t = 1 => t=0.5. Point (1,1).
    # Right contact: (1,1)
    
    # Backward 0->3: (0,1) to (-1,1).
    # (-1, 1) is above y=x (-1 < 1). No cross.
    # 3->2: (-1,1) to (1,-1). Intersects y=x.
    # p=(-1,1) + t*(2, -2). ( -1+2t, 1-2t ).
    # -1+2t = 1-2t => 4t=2 => t=0.5. Point (0,0).
    # Left contact: (0,0)
    
    refined, contacts = clip_contour_to_substrate(contour, substrate, apex)
    
    assert contacts is not None
    # Verify contacts
    c1, c2 = contacts
    # Sorted by x usually
    # (0,0) and (1,1)
    np.testing.assert_allclose(c1, [0, 0], atol=1e-5)
    np.testing.assert_allclose(c2, [1, 1], atol=1e-5)
    
    # Verify retained points are on "apex side"
    # Apex side: (0,1) vs y=x => y>x.
    # Retained: (0,1), (-1,1). (1,1) is on line. (0,0) is on line.
    for p in refined:
        # Check y >= x - epsilon
        assert p[1] >= p[0] - 1e-5

def test_no_clipping_needed():
    """no clipping needed.
    """
    # All points above line
    contour = np.array([[0, 1], [1, 1], [0.5, 2]])
    substrate = ((-1, 0), (2, 0))
    apex = (0.5, 2)
    
    refined, contacts = clip_contour_to_substrate(contour, substrate, apex)
    
    assert len(refined) == 3
    assert contacts is None
