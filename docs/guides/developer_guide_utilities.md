# Developer Guide: Creating Utility Plugins

This guide explains how to create utility plugins for Menipy's image testing and analysis system.

## Overview

Utility plugins provide standalone image analysis functions accessible via the **Utilities** menu (Ctrl+U). They're useful for:
- Image quality analysis
- Edge detection comparison
- Method selection tests
- Diagnostic tools

## Quick Start

Create a file in `plugins/` directory:

```python
# plugins/my_utilities.py

def my_utility(image):
    """Short description shown in list.
    
    Full docstring shown when selected.
    
    Args:
        image: numpy array (grayscale or BGR)
        
    Returns:
        dict with results, or string message
    """
    import cv2
    import numpy as np
    
    # Your analysis code here
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    
    return {
        "mean_intensity": float(np.mean(gray)),
        "std_intensity": float(np.std(gray)),
    }

# Register with menipy
try:
    from menipy.common.registry import register_utility
    register_utility("my_utility", my_utility)
except ImportError:
    pass
```

## Function Signature

```python
def utility_function(image: np.ndarray) -> dict | str
```

- **Input**: NumPy array (BGR or grayscale)
- **Output**: Dictionary (displayed as key-value pairs) or string message

## Registration

Use `register_utility(name, function)`:

```python
from menipy.common.registry import register_utility
register_utility("my_utility", my_utility)
```

## Built-in Utilities

| Utility | Description |
|---------|-------------|
| `image_quality` | Analyzes contrast, uniformity, sharpness, noise, bimodal separation |
| `edge_comparison` | Compares Canny, Otsu, Adaptive methods and recommends best |

## Accessing from Menu

1. Load an image in any experiment window
2. Go to **Utilities → Image Utilities** (Ctrl+U)
3. Select utility from list
4. Click **Run Utility**

## Example: Custom Quality Metric

```python
def focus_quality(image):
    """Measure image focus using Laplacian variance."""
    import cv2
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance = laplacian.var()
    
    return {
        "laplacian_variance": variance,
        "focus_rating": "Sharp" if variance > 100 else "Blurry"
    }
```
