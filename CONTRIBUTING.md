# Contributing to Menipy

Thank you for your interest in contributing to Menipy! This guide will help you get started with development.

## Table of Contents

- [Development Setup](#development-setup)
- [Running Tests](#running-tests)
- [Code Style](#code-style)
- [Adding New Features](#adding-new-features)
- [Submitting Changes](#submitting-changes)

---

## Development Setup

### Prerequisites

- Python 3.10 or higher
- Git

### Quick Setup (Windows PowerShell)

```powershell
# Clone the repository
git clone https://github.com/carenaudo/Menipy.git
cd Menipy

# Create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Upgrade pip and install the package with dev dependencies
python -m pip install --upgrade pip
pip install -e .[dev,test]

# Install pre-commit hooks
pre-commit install
```

### Quick Setup (Linux/macOS)

```bash
# Clone the repository
git clone https://github.com/carenaudo/Menipy.git
cd Menipy

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# Upgrade pip and install the package with dev dependencies
python -m pip install --upgrade pip
pip install -e ".[dev,test]"

# Install pre-commit hooks
pre-commit install
```

---

## Running Tests

### Run all tests

```powershell
pytest
```

### Run tests with coverage

```powershell
pytest --cov=src --cov-report=term-missing
```

### Run specific test file

```powershell
pytest tests/test_geometry.py
```

### Run tests matching a pattern

```powershell
pytest -k "test_edge"
```

---

## Code Style

We use automated tools to maintain consistent code style. All tools are configured in `pyproject.toml`.

### Pre-commit Hooks

Pre-commit hooks run automatically on every commit:

```powershell
# Run all hooks manually
pre-commit run --all-files
```

### Individual Tools

```powershell
# Format code with Black
black .

# Sort imports with isort
isort .

# Lint with Ruff (auto-fix)
ruff check --fix .

# Type check with Mypy
mypy src
```

### Code Style Guidelines

- **Line length**: 88 characters (Black default)
- **Imports**: Sorted by isort (Black-compatible profile)
- **Type hints**: Encouraged for public APIs
- **Docstrings**: Use NumPy-style docstrings (see below)

### Docstring Standards

All public modules, classes, and functions **must have docstrings** following the **NumPy style** convention. This enables:
- Sphinx documentation generation with Napoleon extension
- IDE autocompletion and type hints
- Automated API documentation
- Better code readability and maintainability

#### Module-Level Docstrings

Every Python file should start with a module docstring:

```python
"""Brief module description.

Extended description explaining the module's purpose, main components,
and any important usage notes.
"""

import ...
```

#### Function Docstrings

All public functions must have comprehensive docstrings with:
- **Brief one-line summary** (imperative mood)
- **Extended description** (if needed)
- **Parameters section** (all arguments)
- **Returns section** (return value and type)
- **Raises section** (exceptions that may be raised)
- **Examples section** (usage examples for complex functions)
- **Notes section** (implementation details, warnings)
- **See Also section** (related functions)

**Example:**

```python
def calculate_contact_angle(drop_image, substrate_y, apex_point):
    """Calculate contact angle from sessile drop image.
    
    This function analyzes the drop profile to compute the contact angle
    using the baseline method. The drop must be clearly visible and well-isolated
    from the background.
    
    Parameters
    ----------
    drop_image : ndarray
        Input image containing the sessile drop, shape (H, W) or (H, W, 3).
    substrate_y : int
        Y-coordinate of the substrate baseline in pixels.
    apex_point : tuple of int
        Apex point coordinates as (x, y).
    
    Returns
    -------
    float
        Contact angle in degrees [0, 180].
    
    Raises
    ------
    ValueError
        If substrate_y is outside image bounds or apex_point is invalid.
    TypeError
        If drop_image is not a NumPy array.
    
    Examples
    --------
    >>> import cv2
    >>> img = cv2.imread('drop.png', 0)
    >>> angle = calculate_contact_angle(img, substrate_y=480, apex_point=(320, 200))
    >>> print(f"Contact angle: {angle:.1f}°")
    Contact angle: 102.3°
    
    Notes
    -----
    The substrate must be a clear horizontal line for accurate measurements.
    Substrate_y should be at least 10 pixels from the image bottom.
    
    See Also
    --------
    detect_substrate : Detects substrate baseline
    detect_apex : Finds drop apex point
    """
    # Implementation
    pass
```

#### Class Docstrings

Classes should include:
- Brief description
- Attributes section (for public attributes)
- Methods (auto-documented if they have docstrings)

```python
class DropAnalyzer:
    """Analyzer for sessile drop images.
    
    This class provides methods for detecting and measuring sessile drops,
    including contour detection, substrate identification, and contact angle
    calculation.
    
    Attributes
    ----------
    image : ndarray
        Current image being analyzed.
    substrate_y : int or None
        Detected substrate Y-coordinate.
    drop_contour : ndarray or None
        Detected drop contour points.
    """
    
    def __init__(self, image):
        """Initialize analyzer with image."""
        pass
```

#### Validation

Docstrings are validated using `pydocstyle` with NumPy convention:

```powershell
# Check docstrings in your module
pydocstyle --convention=numpy src/your_module.py

# Check entire codebase
pydocstyle --convention=numpy src/
```

Fix any reported issues before submitting a PR. Common issues:
- `D100`: Missing module docstring
- `D101`: Missing class docstring  
- `D102`: Missing public method docstring
- `D103`: Missing function docstring
- `D200`: One-liner docstring on multiple lines

---

## Adding New Features

### Adding a New Pipeline

Pipelines are the core analysis workflows in Menipy. To add a new pipeline:

1. Create a new directory under `src/menipy/pipelines/your_pipeline/`
2. Implement the required stages (see existing pipelines for reference)
3. Register your pipeline in the `__init__.py`

📖 **Full guide**: [Adding a New Analysis Pipeline](docs/guides/developer_guide_pipelines.md)

### Adding a New Plugin

Plugins extend Menipy with custom filters, solvers, and other functionality:

1. Create your plugin in the `plugins/` directory
2. Implement the required interface for your plugin type
3. Register using the `@register` decorator

📖 **Full guide**: [Adding a New Image Filter Plugin](docs/guides/developer_guide_plugins.md)

---

## Submitting Changes

### Workflow

1. **Fork** the repository on GitHub
2. **Create a branch** from `main`:
   ```powershell
   git checkout -b feature/your-feature-name
   ```
3. **Make your changes** following the code style guidelines
4. **Run tests** to ensure nothing is broken:
   ```powershell
   pytest
   pre-commit run --all-files
   ```
5. **Commit** with a descriptive message:
   ```powershell
   git commit -m "feat: add new edge detection algorithm"
   ```
6. **Push** your branch and open a Pull Request

### Commit Message Convention

We follow [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation changes
- `style:` - Code style changes (formatting, etc.)
- `refactor:` - Code refactoring
- `test:` - Adding or updating tests
- `chore:` - Maintenance tasks

### Pull Request Checklist

Before submitting a PR, ensure:

- [ ] All tests pass (`pytest`)
- [ ] Code is formatted (`black .`)
- [ ] Imports are sorted (`isort .`)
- [ ] No linting errors (`ruff check .`)
- [ ] Documentation is updated if needed
- [ ] Commit messages follow convention

---

## Getting Help

- **Issues**: [GitHub Issues](https://github.com/carenaudo/Menipy/issues)
- **Discussions**: Open an issue for questions or feature ideas

We appreciate all contributions, from bug reports to documentation improvements!
