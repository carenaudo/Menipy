# SOLID and DRY Principles Analysis

This document provides a comprehensive analysis of the Menipy codebase for adherence to **SOLID** and **DRY** (Don't Repeat Yourself) principles, along with specific improvement recommendations.

---

## Executive Summary

| Principle | Current State | Rating |
|-----------|--------------|--------|
| **S** - Single Responsibility | Good in models, mixed in GUI | ⭐⭐⭐☆☆ |
| **O** - Open/Closed | Strong with plugin system | ⭐⭐⭐⭐☆ |
| **L** - Liskov Substitution | Good in pipelines | ⭐⭐⭐⭐☆ |
| **I** - Interface Segregation | Needs improvement | ⭐⭐☆☆☆ |
| **D** - Dependency Inversion | Partial implementation | ⭐⭐⭐☆☆ |
| **DRY** - Don't Repeat Yourself | Significant violations | ⭐⭐☆☆☆ |

---

## SOLID Principles Analysis

### S - Single Responsibility Principle (SRP)

> *A class should have only one reason to change.*

#### ✅ Good Examples

| File | Why It's Good |
|------|---------------|
| [context.py](file:///d:/programacion/Menipy/src/menipy/models/context.py) | Single responsibility: state container for pipeline data |
| [config.py](file:///d:/programacion/Menipy/src/menipy/models/config.py) | Each settings class handles one configuration aspect |
| [acquisition.py](file:///d:/programacion/Menipy/src/menipy/common/acquisition.py) | Focused solely on image loading |

#### ❌ Violations

**1. `MainController` (756 lines)**

[main_controller.py](file:///d:/programacion/Menipy/src/menipy/gui/main_controller.py) handles too many responsibilities:
- Pipeline execution coordination
- Menu action handling  
- Dialog management
- Camera frame processing
- ROI selection handling
- Preprocessing preview
- Preview panel updates

**Improvement:**
```python
# Split into focused controllers:
class DialogController:
    """Handles dialog lifecycle and configuration."""
    
class SourceController:
    """Manages camera/file source selection."""
    
class PreviewController:
    """Handles preview panel updates and ROI selection."""
```

**2. `PipelineController` (592 lines)**

[pipeline_controller.py](file:///d:/programacion/Menipy/src/menipy/gui/controllers/pipeline_controller.py) manages:
- Pipeline execution
- Results display
- Log management
- Measurement tracking
- Stage configuration

---

### O - Open/Closed Principle (OCP)

> *Software entities should be open for extension, but closed for modification.*

#### ✅ Good Examples

**Plugin System Architecture**

The plugin system in [registry.py](file:///d:/programacion/Menipy/src/menipy/common/registry.py) and [plugin_db.py](file:///d:/programacion/Menipy/src/menipy/common/plugin_db.py) allows adding new:
- Edge detectors
- Solvers
- Preprocessors
- Overlayers

Without modifying core code.

**Template Method Pattern in Pipelines**

[base.py](file:///d:/programacion/Menipy/src/menipy/pipelines/base.py) uses template method pattern:
```python
class PipelineBase:
    def run(self, **kwargs):
        # Fixed algorithm
        self._call_stage(ctx, "acquisition", self.do_acquisition)
        self._call_stage(ctx, "preprocessing", self.do_preprocessing)
        # ... extensible via overriding do_* methods
```

#### ❌ Violations

**Edge Detection Method Selection**

[edge_detection.py](file:///d:/programacion/Menipy/src/menipy/common/edge_detection.py#L259-L298) uses a long `if-elif` chain:

```python
# Current (violates OCP - adding new method requires modifying this file)
if settings.method == "canny":
    edges = cv2.Canny(...)
elif settings.method == "threshold":
    _, edges = cv2.threshold(...)
elif settings.method == "sobel":
    # ...
```

**Improvement:**
```python
# Use strategy pattern with registry
EDGE_METHODS = {
    "canny": CannyEdgeDetector(),
    "sobel": SobelEdgeDetector(),
    "threshold": ThresholdEdgeDetector(),
}

def run(ctx, settings):
    detector = EDGE_METHODS.get(settings.method, CannyEdgeDetector())
    return detector.detect(ctx, settings)
```

---

### L - Liskov Substitution Principle (LSP)

> *Subtypes must be substitutable for their base types.*

#### ✅ Good Examples

All pipeline implementations properly extend `PipelineBase`:
- [PendantPipeline](file:///d:/programacion/Menipy/src/menipy/pipelines/pendant/stages.py)
- [SessilePipeline](file:///d:/programacion/Menipy/src/menipy/pipelines/sessile/stages.py)
- [CapillaryRisePipeline](file:///d:/programacion/Menipy/src/menipy/pipelines/capillary_rise/stages.py)
- [CaptiveBubblePipeline](file:///d:/programacion/Menipy/src/menipy/pipelines/captive_bubble/stages.py)

Each can be substituted in `pipeline.run()` calls.

#### ⚠️ Minor Issues

**Inconsistent `geometry` attribute types:**
- `CapillaryRisePipeline.do_geometry()` sets `ctx.geometry` to a `dict`
- `CaptiveBubblePipeline.do_geometry()` sets it to `CaptiveBubbleGeometry`

This requires type checking in downstream code.

---

### I - Interface Segregation Principle (ISP)

> *Clients should not be forced to depend on interfaces they don't use.*

#### ❌ Significant Violations

**1. `Context` Class - Fat Data Object**

[context.py](file:///d:/programacion/Menipy/src/menipy/models/context.py) has 40+ fields:
```python
class Context(BaseModel):
    frames: Any | None = None
    current_frame: Optional[Frame] = None
    image_path: Optional[str] = None
    # ... 40+ more fields
```

Not all pipelines use all fields, but all must carry this bloat.

**Improvement:**
```python
# Use composition with focused contexts
class AcquisitionContext(BaseModel):
    frames: list[Frame]
    image_path: Optional[str]
    camera_id: Optional[int]

class EdgeDetectionContext(BaseModel):
    contour: Optional[Contour]
    fluid_interface_contour: Optional[Contour]

class PipelineContext(BaseModel):
    """Compose only what's needed"""
    acquisition: AcquisitionContext
    edge_detection: Optional[EdgeDetectionContext] = None
```

**2. `EdgeDetectionSettings` - Monolithic Configuration**

[config.py](file:///d:/programacion/Menipy/src/menipy/models/config.py#L114-L163) packs ALL method parameters:
```python
class EdgeDetectionSettings(BaseModel):
    canny_threshold1: int  # Only for Canny
    sobel_kernel_size: int  # Only for Sobel
    laplacian_kernel_size: int  # Only for Laplacian
    active_contour_iterations: int  # Only for Active Contour
```

**Improvement:**
```python
class CannySettings(BaseModel):
    threshold1: int = 50
    threshold2: int = 150
    aperture_size: Literal[3, 5, 7] = 3

class EdgeDetectionSettings(BaseModel):
    method: str = "canny"
    canny: CannySettings = Field(default_factory=CannySettings)
    sobel: Optional[SobelSettings] = None
    # Each method has its own focused settings
```

---

### D - Dependency Inversion Principle (DIP)

> *High-level modules should not depend on low-level modules. Both should depend on abstractions.*

#### ✅ Good Examples

**Registry-based Plugin Loading**

High-level code depends on abstractions:
```python
# Pipeline uses registry, not concrete implementations
from .registry import EDGE_DETECTORS
detector = EDGE_DETECTORS.get("canny")
```

#### ❌ Violations

**1. Direct Module Loading in Pipelines**

[pendant/stages.py](file:///d:/programacion/Menipy/src/menipy/pipelines/pendant/stages.py#L14-L21):
```python
# Direct dependency on file path - hard to test
_repo_root = Path(__file__).resolve().parents[4]
_adsa_path = _repo_root / "plugins" / "young_laplace_adsa.py"
_adsa_mod = _load_module_from_path(_adsa_path, "adsa_plugins.young_laplace_adsa")
```

This is repeated in every pipeline file.

**Improvement:**
```python
# Use dependency injection
class PendantPipeline(PipelineBase):
    def __init__(self, solver: Callable = None, **kwargs):
        super().__init__(**kwargs)
        self.solver = solver or registry.SOLVERS.get("young_laplace")
```

---

## DRY Principle Analysis

> *Every piece of knowledge must have a single, unambiguous, authoritative representation.*

### ❌ Major Violations

#### 1. Repetitive Registry Functions

[registry.py](file:///d:/programacion/Menipy/src/menipy/common/registry.py) has 11 nearly identical functions:

```python
def register_edge(name: str, fn: Callable[..., Any]) -> None:
    EDGE_DETECTORS[name] = fn

def register_solver(name: str, fn: Callable[..., Any]) -> None:
    SOLVERS[name] = fn

def register_acquisition(name: str, fn: Callable[..., Any]) -> None:
    ACQUISITIONS[name] = fn
# ... 8 more identical functions
```

**Improvement:**
```python
class Registry:
    def __init__(self, name: str):
        self.name = name
        self._items: Dict[str, Callable] = {}
    
    def register(self, name: str, fn: Callable):
        self._items[name] = fn
    
    def get(self, name: str) -> Optional[Callable]:
        return self._items.get(name)

# Usage
EDGE_DETECTORS = Registry("edge_detectors")
SOLVERS = Registry("solvers")
ACQUISITIONS = Registry("acquisitions")
```

---

#### 2. Duplicated `_ensure_contour` Function

This helper function is defined identically in **4 pipeline files**:

| File | Lines |
|------|-------|
| [pendant/stages.py](file:///d:/programacion/Menipy/src/menipy/pipelines/pendant/stages.py#L33-L76) | 33-76 |
| [sessile/stages.py](file:///d:/programacion/Menipy/src/menipy/pipelines/sessile/stages.py#L27-L46) | 27-46 |
| [capillary_rise/stages.py](file:///d:/programacion/Menipy/src/menipy/pipelines/capillary_rise/stages.py#L22-L27) | 22-27 |
| [captive_bubble/stages.py](file:///d:/programacion/Menipy/src/menipy/pipelines/captive_bubble/stages.py#L18-L25) | 18-25 |

**Improvement:**
Move to base class or common module:
```python
# In menipy/pipelines/base.py
def ensure_contour(ctx: Context, settings: EdgeDetectionSettings) -> np.ndarray:
    """Get contour from context, running edge detection if needed."""
    if ctx.contour and hasattr(ctx.contour, "xy"):
        return np.asarray(ctx.contour.xy, dtype=float)
    edged.run(ctx, settings=settings or EdgeDetectionSettings())
    return np.asarray(ctx.contour.xy, dtype=float)
```

---

#### 3. Duplicated Plugin Loading Boilerplate

Every pipeline file has:
```python
from menipy.common.plugins import _load_module_from_path

_repo_root = Path(__file__).resolve().parents[4]
_toy_path = _repo_root / "plugins" / "toy_young_laplace.py"
_toy_mod = _load_module_from_path(_toy_path, "adsa_plugins.toy_young_laplace")
young_laplace_sphere = getattr(_toy_mod, "toy_young_laplace")
```

**Improvement:**
```python
# In menipy/common/plugin_loader.py
def get_solver(name: str) -> Callable:
    """Load solver from plugins or registry."""
    if name in registry.SOLVERS:
        return registry.SOLVERS[name]
    return load_plugin_solver(name)

# Usage in pipeline
from menipy.common.plugin_loader import get_solver
solver = get_solver("young_laplace")
```

---

#### 4. Repeated Edge Detection Method Handling

[edge_detection.py](file:///d:/programacion/Menipy/src/menipy/common/edge_detection.py#L259-L297) repeats the pattern:
```python
if settings.method == "X":
    edges = cv2.X(...)
    xy = _edges_to_xy(edges, settings.min_contour_length, settings.max_contour_length)
```

Same structure repeated 6 times.

**Improvement:**
```python
class EdgeMethod:
    def detect(self, img: np.ndarray, settings: EdgeDetectionSettings) -> np.ndarray:
        raise NotImplementedError

class CannyMethod(EdgeMethod):
    def detect(self, img, settings):
        return cv2.Canny(img, settings.canny_threshold1, settings.canny_threshold2,
                        apertureSize=settings.canny_aperture_size)

class SobelMethod(EdgeMethod):
    def detect(self, img, settings):
        grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=settings.sobel_kernel_size)
        grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=settings.sobel_kernel_size)
        magnitude = cv2.magnitude(grad_x, grad_y)
        magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        _, edges = cv2.threshold(magnitude, settings.threshold_value, 255, cv2.THRESH_BINARY)
        return edges

METHODS = {"canny": CannyMethod(), "sobel": SobelMethod(), ...}
```

---

#### 5. Duplicated Overlay Drawing Logic

Both [captive_bubble/stages.py](file:///d:/programacion/Menipy/src/menipy/pipelines/captive_bubble/stages.py#L109-L122) and other pipelines duplicate measurement number drawing:

```python
# Repeated in multiple pipelines
if hasattr(ctx, 'measurement_sequence') and ctx.measurement_sequence is not None:
    import cv2
    measurement_text = f"Measurement #{ctx.measurement_sequence}"
    # ... drawing code
```

**Improvement:**
```python
# In menipy/common/overlay.py
def draw_measurement_number(img: np.ndarray, sequence: int) -> np.ndarray:
    """Draw measurement sequence number on image."""
    # Centralized implementation
```

---

## Priority Improvement Roadmap

### Phase 1: Quick Wins (Low Risk, High Impact)

| Task | Files Affected | Effort |
|------|---------------|--------|
| Extract `_ensure_contour` to base | 4 pipeline files | 1 hour |
| Centralize plugin loading | All pipelines | 2 hours |
| Refactor registry to use classes | `registry.py` | 1 hour |

### Phase 2: Medium-Term (Moderate Risk)

| Task | Files Affected | Effort |
|------|---------------|--------|
| Strategy pattern for edge detection | `edge_detection.py` | 4 hours |
| Split `MainController` | GUI module | 8 hours |
| Compose smaller context objects | `models/`, all pipelines | 6 hours |

### Phase 3: Long-Term (Higher Risk)

| Task | Files Affected | Effort |
|------|---------------|--------|
| Full dependency injection | All pipelines | 16 hours |
| Interface segregation for settings | `models/config.py`, dialogs | 8 hours |

---

## Metrics Summary

| Category | Count |
|----------|-------|
| Major DRY violations | **5** |
| SRP violations | **2** (large controllers) |
| OCP violations | **1** (edge detection) |
| ISP violations | **2** (Context, Settings) |
| DIP violations | **1** (plugin loading) |
| Files needing refactoring | ~15 |
| Estimated total effort | ~40 hours |

---

## Conclusion

The Menipy codebase has a **solid architectural foundation** with:
- ✅ Good use of template method pattern in `PipelineBase`
- ✅ Extensible plugin system via registries
- ✅ Well-structured configuration models (Pydantic)

**Key areas for improvement:**
1. **DRY violations** are the most pressing - duplicated code across pipelines
2. **Large controller classes** need decomposition for maintainability
3. **Interface segregation** would reduce coupling between components

Addressing Phase 1 issues first will yield the highest return on investment and reduce maintenance burden significantly.
