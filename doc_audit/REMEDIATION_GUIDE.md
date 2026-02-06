# Detailed Remediation Guide (HIGH Priority)

This guide provides specific recommendations for the 22 HIGH-priority files.
All example docstrings use NumPy style as recommended by the project (Sphinx + napoleon).

## Docstring Style Reference

### NumPy Style Format

```python
def example_function(param1, param2):
    """Brief one-line description.
    
    Extended description can span multiple lines and explain
    the purpose, behavior, and any important notes.
    
    Parameters
    ----------
    param1 : type
        Description of param1.
    param2 : type, optional
        Description of param2. Default is None.
    
    Returns
    -------
    type
        Description of return value.
    
    Raises
    ------
    ValueError
        When param1 is invalid.
    
    Examples
    --------
    >>> result = example_function(10, 20)
    >>> print(result)
    30
    
    Notes
    -----
    Important implementation notes here.
    
    See Also
    --------
    related_function : Related functionality.
    """
    pass
```

---

## File-by-File Remediation Details



### 1. plugins\auto_adaptive_edge.py

**Status**: 1 functions, 0 classes missing docstrings


**Undocumented Functions** (1):


- `auto_adaptive_detect(img, settings, otsu_variance_threshold, illumination_cv_threshold, gradient_strength_threshold, canny_low, canny_high, adaptive_block_size, adaptive_c, min_contour_length, max_contour_length)`




### 2. plugins\bezier_edge.py

**Status**: 1 functions, 0 classes missing docstrings


**Undocumented Functions** (1):


- `bezier_like(img)`




### 3. plugins\circle_edge.py

**Status**: 1 functions, 0 classes missing docstrings


**Undocumented Functions** (1):


- `_fallback_circle(h, w, points)`




### 4. plugins\detect_apex.py

**Status**: 3 functions, 0 classes missing docstrings


**Undocumented Functions** (3):


- `detect_apex_pendant(drop_contour)`

- `detect_apex_sessile(drop_contour, substrate_y)`

- `detect_apex_auto(drop_contour, pipeline)`




### 5. plugins\detect_drop.py

**Status**: 3 functions, 0 classes missing docstrings


**Undocumented Functions** (3):


- `model_post_init(__context)`

- `detect_drop_sessile(image, clahe_clip_limit, clahe_tile_size, int], 8)`

- `detect_drop_pendant(image, min_area_fraction)`




### 6. plugins\detect_needle.py

**Status**: 3 functions, 0 classes missing docstrings


**Undocumented Functions** (3):


- `model_post_init(__context)`

- `detect_needle_sessile(image, clahe_clip_limit, # Keeps kwargs for backward compatibility
    clahe_tile_size, int], 8)`

- `detect_needle_pendant(image, drop_contour, tolerance)`




### 7. plugins\detect_roi.py

**Status**: 3 functions, 0 classes missing docstrings


**Undocumented Functions** (3):


- `detect_roi_sessile(image, drop_contour, substrate_y, needle_rect, int, int, int]], padding)`

- `detect_roi_pendant(image, drop_contour, apex_point, int]], padding)`

- `detect_roi_auto(image, pipeline)`




### 8. plugins\detect_substrate.py

**Status**: 2 functions, 0 classes missing docstrings


**Undocumented Functions** (2):


- `detect_substrate_gradient(image, clahe_clip_limit, clahe_tile_size, int], 8)`

- `detect_substrate_hough(image, canny_low, canny_high, hough_threshold, angle_tolerance)`




### 9. plugins\sine_edge.py

**Status**: 1 functions, 0 classes missing docstrings


**Undocumented Functions** (1):


- `_fallback_sine(h, w, waves, amplitude, points)`




### 10. plugins\young_laplace_adsa.py

**Status**: 4 functions, 0 classes missing docstrings


**Undocumented Functions** (4):


- `integrate_profile(R0_mm, beta, settings)`

- `event_detach(s, y, beta)`

- `young_laplace_adsa(params, physics, geometry)`

- ... and 1 more




### 11. src\menipy\common\plugins.py

**Status**: 3 functions, 0 classes missing docstrings


**Undocumented Functions** (3):


- `_load_module_from_path(path, module_name)`

- `_load_plugin(row)`

- `discover_and_load_from_db(db, settings_key)`




### 12. plugins\output_json.py

**Status**: 1 functions, 0 classes missing docstrings


**ACTION: Add Module Docstring**


Add this at the very top of the file:


```python

"""Brief module purpose.


This module provides [detailed description],

including [main components/responsibilities].

"""

```



**Undocumented Functions** (1):


- `output_results_json(ctx)`




### 13. plugins\overlayer_simple.py

**Status**: 1 functions, 0 classes missing docstrings


**ACTION: Add Module Docstring**


Add this at the very top of the file:


```python

"""Brief module purpose.


This module provides [detailed description],

including [main components/responsibilities].

"""

```



**Undocumented Functions** (1):


- `add_simple_overlay(ctx)`




### 14. plugins\physics_dummy.py

**Status**: 1 functions, 0 classes missing docstrings


**ACTION: Add Module Docstring**


Add this at the very top of the file:


```python

"""Brief module purpose.


This module provides [detailed description],

including [main components/responsibilities].

"""

```



**Undocumented Functions** (1):


- `dummy_physics(ctx)`




### 15. plugins\preproc_blur.py

**Status**: 1 functions, 0 classes missing docstrings


**ACTION: Add Module Docstring**


Add this at the very top of the file:


```python

"""Brief module purpose.


This module provides [detailed description],

including [main components/responsibilities].

"""

```



**Undocumented Functions** (1):


- `blur_preprocessor(ctx)`




---

## General Remediation Checklist


For each HIGH-priority file:


- [ ] Add module-level docstring (3-5 sentences)

- [ ] Add one-line docstrings to all public functions/classes

- [ ] For complex functions (10+ lines), add parameter/return docs

- [ ] Explain any magic numbers with inline comments

- [ ] Remove or explain large commented-out code blocks

- [ ] Ensure type hints on function signatures


## Coverage Targets After Remediation


- **Current avg coverage**: 47.69%

- **Target HIGH priority**: 85%+

- **Target MEDIUM priority**: 70%+

- **Target LOW priority**: 60%+
