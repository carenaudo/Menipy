# Developer Quick Start Guide - NumPy Docstrings & Pre-commit

## 🎯 What Changed?

Your project now requires **NumPy-style docstrings** validated automatically through **pre-commit hooks**. This ensures consistent, high-quality documentation across the codebase.

---

## 💻 First-Time Setup (5 minutes)

### Step 1: Install Pre-commit Hooks
```powershell
# One-time setup
pip install pre-commit
pre-commit install

# Verify installation
pre-commit --version
```

### Step 2: Try It Out
```powershell
# This will run all hooks on your changes
git add .
git commit -m "test: first commit with pre-commit"
```

**What happens**: Hooks run automatically before commit. If any fail, fix and re-commit.

---

## 📝 Writing Docstrings

### Module Level (Required)
Add to the top of every Python file:
```python
"""Brief description of what this module does.

Extended explanation of key components, main functions,
and usage examples if needed.
"""

import sys
```

### Function Level (Required for Public APIs)
```python
def analyze_drop(image, settings):
    """Analyze sessile drop from image to extract contact angle.
    
    This function detects the drop profile and calculates the contact
    angle using the baseline method.
    
    Parameters
    ----------
    image : ndarray
        RGB or grayscale image, shape (H, W, 3) or (H, W).
    settings : AnalysisSettings
        Configuration for the analysis.
    
    Returns
    -------
    dict
        Results containing:
        - 'contact_angle': float, in degrees
        - 'confidence': float, 0-1 score
        - 'debug_image': ndarray (if requested)
    
    Raises
    ------
    ValueError
        If image format is not recognized.
    TypeError
        If settings is not AnalysisSettings.
    
    Examples
    --------
    >>> import cv2
    >>> img = cv2.imread('drop.png')
    >>> settings = AnalysisSettings(method='baseline')
    >>> result = analyze_drop(img, settings)
    >>> print(f"Angle: {result['contact_angle']:.1f}°")
    Angle: 102.3°
    
    Notes
    -----
    Substrate detection is critical for accuracy. Ensure the substrate
    is clearly visible as a horizontal line.
    
    See Also
    --------
    detect_substrate : Find substrate baseline
    detect_apex : Find drop apex
    """
```

### Class Level (Recommended)
```python
class DropAnalyzer:
    """Analyzes sessile drop images for surface tension measurement.
    
    This class provides methods for complete drop analysis, from detection
    through contact angle calculation.
    
    Attributes
    ----------
    image : ndarray or None
        Current image being analyzed.
    substrate_y : int or None
        Detected substrate Y-coordinate.
    """
    
    def __init__(self, image=None):
        """Initialize analyzer with optional image."""
        pass
```

---

## ✅ Validation

### Automatic (Pre-commit)
Your docstrings are checked **before each commit**:
```
Running hook: pydocstyle ...................... PASS
```

### Manual (Anytime)
```powershell
# Check one file
pydocstyle --convention=numpy src/menipy/analysis/drop.py

# Check entire directory
pydocstyle --convention=numpy src/

# Fix suggestions shown if any issues found
# Fix them, then re-commit
```

### Common Issues
| Error | Fix |
|-------|-----|
| `D100` | Add module docstring at top of file |
| `D101` | Add docstring to class |
| `D102` | Add docstring to public method |
| `D103` | Add docstring to function |
| `D200` | Split one-liner docstring to multiple lines |

---

## 🔄 Typical Workflow

```powershell
# 1. Create branch
git checkout -b feature/my-feature

# 2. Make changes
# - Write new functions/classes
# - Add docstrings following NumPy style
# - Commit changes

# 3. Stage changes
git add .

# 4. Commit (hooks run automatically!)
git commit -m "feat: add new image filter"
# If hooks fail:
#   - Review error messages
#   - Fix docstrings
#   - Stage again
#   - Commit again

# 5. Push branch
git push origin feature/my-feature

# 6. Open PR on GitHub
```

---

## 🚫 Bypassing Hooks (Emergency Only!)

```powershell
# NEVER do this in normal workflow!
# Use only for urgent fixes that need immediate deployment
git commit --no-verify

# After using --no-verify:
# 1. MUST fix docstrings immediately
# 2. Create follow-up commit with proper docs
# 3. Inform team in PR description
```

---

## 📚 Reference Resources

### In This Project
- **[CONTRIBUTING.md](CONTRIBUTING.md)** - Full docstring standards guide
- **[DOCUMENTATION_PROJECT_SUMMARY.md](DOCUMENTATION_PROJECT_SUMMARY.md)** - Project overview
- **[PHASE_1_SUMMARY.md](PHASE_1_SUMMARY.md)** - Detailed detection plugin docs

### External
- [NumPy Docstring Style](https://numpydoc.readthedocs.io/en/latest/format.html)
- [pydocstyle Documentation](http://www.pydocstyle.org/)
- [Pre-commit Framework](https://pre-commit.com/)

---

## ❓ FAQ

**Q: Do internal/private functions need docstrings?**  
A: Only public functions (not starting with `_`). Private functions are optional but recommended.

**Q: Can I use different docstring format?**  
A: No, NumPy is the project standard. All docstrings must follow it.

**Q: What if I'm adding code to existing file with no module docstring?**  
A: Add the module docstring at the top. You'll only see it on first commit, then it passes.

**Q: How do I add examples to docstrings?**  
A: Use the Examples section. Include `>>>` for Python code that can be tested.

**Q: Do old files need docstring updates?**  
A: Only when you modify them. New files must have complete docs.

**Q: What's the performance impact of validation?**  
A: ~1-2 seconds per commit (background check). Minimal overhead.

---

## 🆘 Getting Help

### Before Asking:
1. Check `pydocstyle --help`
2. Review CONTRIBUTING.md Docstring Standards section
3. Look at similar existing functions in codebase

### Then Ask:
- **GitHub Issues**: Technical questions or bulk problems
- **PR Comments**: Specific review feedback
- **Team Chat**: Quick clarification questions

---

## 🎓 Training by Example

### Example 1: Missing Module Docstring
**Before:**
```python
import cv2
import numpy as np

def detect_edge(image):
    ...
```

**Error:** `D100 Missing docstring in public module`

**After:**
```python
"""Image edge detection utilities using OpenCV.

Provides high-level API for common edge detection algorithms
with automatic parameter tuning.
"""

import cv2
import numpy as np

def detect_edge(image):
    ...
```

---

### Example 2: Missing Function Docstring
**Error:** `D103 Missing docstring in public function`

**Before:**
```python
def calculate_contact_angle(drop_coords):
    # Calculate angle from drop coordinates
    return angle
```

**After:**
```python
def calculate_contact_angle(drop_coords):
    """Calculate contact angle from drop coordinate points.
    
    Parameters
    ----------
    drop_coords : ndarray
        Contour points of drop, shape (N, 2).
    
    Returns
    -------
    float
        Contact angle in degrees [0, 180].
    """
    return angle
```

---

## ✨ Best Practices

✅ **DO:**
- Add docstrings as you write code (not after)
- Include practical examples for complex functions
- Document edge cases and limitations in Notes
- Use clear, professional language
- Make parameter descriptions concise but complete

❌ **DON'T:**
- Write vague docstrings ("Do stuff")
- Document only return types, ignore parameters
- Add docstrings to test functions (test_ prefix)
- Copy-paste docstring templates without customization
- Leave placeholder docstrings (pre-commit will catch them)

---

## 📊 Team Metrics

**Current Project Status:**
- 251 Python files scanned
- 1,844 functions identified
- 56.7% documentation coverage (updated: 51.2% base + improvements)
- **Target: 70% by end of Q2**

Check progress: `python PHASE_3_COMPLETION_REPORT.py`

---

## 🎯 Your Role

As a developer, you're responsible for:
1. ✅ Adding NumPy docstrings to new code
2. ✅ Fixing docstring issues before committing
3. ✅ Requesting reviews for complex functions
4. ✅ Helping team members with docstring questions

Together, we can reach 70% documentation coverage!

---

**Last Updated**: Phase 3 Complete  
**Status**: Active & Enforced  
**Questions?** Check CONTRIBUTING.md or open an issue
