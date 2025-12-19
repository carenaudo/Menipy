# SOLID and DRY Principles Analysis (Gemini Report)

This document presents an analysis of the **Menipy** codebase (`src/menipy`) specifically evaluating adherence to **SOLID** and **DRY** principles.

**Date:** 2025-12-19
**Evaluator:** Gemini Agent

---

## 1. Executive Summary

The codebase demonstrates a strong foundation with clear modularity in some areas (Pipelines, Pydantic models), but suffers from significant growing pains common in research software.

*   **Overall Architecture:** Plugin-based architecture (Good OCP foundation).
*   **Main Weaknesses:**
    *   **DRY:** High duplication in boilerplate code (plugin loading, registry).
    *   **SRP:** "God classes" in GUI controllers (`MainController`).
    *   **DIP:** Hardcoded file paths for plugin loading bypass true dependency injection.

---

## 2. SOLID Principles Analysis

### S - Single Responsibility Principle (SRP)
*A class should have one, and only one, reason to change.*

*   **✅ Strengths:**
    *   **Models:** `menipy.models` classes (e.g., `PhysicsParams`) are well-scoped data containers.
    *   **Acquisition:** `menipy.common.acquisition` focuses solely on loading images.

*   **❌ Violations:**
    *   **`MainController` (`src/menipy/gui/main_controller.py`)**: This class has become a "God Object" (750+ lines). It handles:
        *   Window layout/state saving.
        *   Camera device streaming logic.
        *   Dialog instantiation and execution.
        *   Coordinating pipeline execution.
        *   Wiring signals for every sub-component.
        *   *Impact:* Changing camera logic requires modifying the main controller; changing UI layout does too. High coupling.
    *   **`Context` (`src/menipy/models/context.py`)**: While intended as a data bus, it mixes responsibilities of *input data* (frames, images), *processing state* (contours), *configuration* (settings), and *output* (results, logs). It changes for almost any reason.

### O - Open/Closed Principle (OCP)
*Entities should be open for extension, but closed for modification.*

*   **✅ Strengths:**
    *   **Pipeline Architecture:** `PipelineBase` uses the Template Method pattern. Adding a new stage like `do_overlay` doesn't require changing the `run` method in the base class (mostly).
    *   **Plugin Registries:** The dictionary-based registries allow adding new components without changing the registry definition.

*   **❌ Violations:**
    *   **Edge Detection Selection (`src/menipy/common/edge_detection.py`)**: The `run` function contains a large `if/elif` block checking `settings.method == "canny"`, `settings.method == "sobel"`, etc.
        *   *Impact:* Adding a new edge detector (e.g., "DeepLearningEdge") requires modifying the core `run` function, violating OCP.

### L - Liskov Substitution Principle (LSP)
*Subtypes must be substitutable for their base types.*

*   **✅ Strengths:**
    *   Pipelines (Pendant, Sessile) seem to interchangeably work within the `PipelineController`, respecting the `PipelineBase` contract.

*   **⚠️ Risks:**
    *   Inconsistent typing in `Context.geometry` (sometimes a dictionary, sometimes a `Geometry` object) across different pipelines could lead to runtime errors when identifying the type, breaking substitution.

### I - Interface Segregation Principle (ISP)
*Clients should not be forced to depend on methods/data they do not use.*

*   **❌ Violations:**
    *   **The Global `Context` Object:** Every pipeline stage receives the full `Context` object.
        *   *Impact:* An `acquisition` stage has access to `fit_results` (which don't exist yet) and `overlay` settings. This makes unit testing hard because one must mock a massive object just to test a small function.
    *   **`EdgeDetectionSettings`**: Passes *all* parameters for *all* algorithms (Canny, Sobel, Active Contour) in one flat structure. A Canny detector depends on `sobel_kernel_size` simply because it's in the same object.

### D - Dependency Inversion Principle (DIP)
*Depend on abstractions, not concretions.*

*   **❌ Violations:**
    *   **Hardcoded Plugin Loading (`src/menipy/pipelines/pendant/stages.py`)**:
        ```python
        _repo_root = Path(__file__).resolve().parents[4]
        _adsa_path = _repo_root / "plugins" / "young_laplace_adsa.py"
        ```
        The pipeline depends directly on the *file system structure* and a specific *file location*. It does not rely on an abstract `Solver` interface injected at runtime.
    *   **Module-level Imports of Implementations**: Direct imports of `cv2` inside logic often couple the business logic to the specific implementation library.

---

## 3. DRY (Don't Repeat Yourself) Analysis

*   **❌ Critical Violations:**
    1.  **Registry Boilerplate (`src/menipy/common/registry.py`)**:
        There are ~10 functions (`register_edge`, `register_solver`, etc.) that are identical except for the dictionary they target.
    2.  **Plugin Loading Boilerplate**:
        The code to find `_repo_root` and `_load_module_from_path` is pasted into multiple pipeline files.
    3.  **Contour Validation (`_ensure_contour`)**:
        The helper function `_ensure_contour` appears to be copied into `pendant/stages.py` and likely other pipelines. It contains complex logic for fallback image loading that should be central.
    4.  **Measurement Overlays**:
        Code to draw specific markers and text is repeated across pipeline `do_overlay` methods.

---

## 4. Improvement Plan & Recommendations

### Refactoring Strategy

#### 1. Fix DRY in Registry (Low Effort, High Value)
**Current:**
```python
def register_edge(name, fn): EDGE_DETECTORS[name] = fn
def register_solver(name, fn): SOLVERS[name] = fn
```
**Proposed:**
Create a generic `Registry` class.
```python
class Registry:
    def __init__(self, name: str):
        self._items = {}
    def register(self, key: str, item: Any):
        self._items[key] = item
    def get(self, key: str):
        return self._items.get(key)

# In registry.py
edge_detectors = Registry("edge_detectors")
solvers = Registry("solvers")
```

#### 2. Implement Strategy Pattern for Edge Detection (Fix OCP)
**Current:** `if settings.method == 'canny': ...`
**Proposed:**
Define an abstract base class `EdgeDetectorStrategy`.
```python
class EdgeDetectorStrategy(ABC):
    @abstractmethod
    def detect(self, img: np.ndarray, settings: EdgeDetectionSettings) -> np.ndarray: ...

class CannyDetector(EdgeDetectorStrategy): ...
class SobelDetector(EdgeDetectorStrategy): ...

# In edge_detection.py run():
strategy = registry.edge_detectors.get(settings.method)
if strategy:
    xy = strategy.detect(img, settings)
```

#### 3. Decompose `MainController` (Fix SRP)
Break it down by functional area. simpler controllers can be instantiated by `MainController`.
*   `CameraManager`: Handles all `QCamera` / streaming logic.
*   `LayoutManager`: Handles saving/restoring window geometry.
*   `DialogCoordinator`: Handles showing configuration dialogs.

#### 4. Adoption of Dependency Injection for Plugins (Fix DIP)
Instead of pipelines determining *where* a plugin file lives:
1.  Use a central `PluginLoader` that scans directories at startup and populates the `Registry`.
2.  Pipelines request a solver by name: `solver = registry.solvers.get("young_laplace")`.
3.  Dependencies are injected into the Pipeline constructor if needed.

#### 5. Centralize Pipeline Utilities
Move `_ensure_contour` to `menipy.pipelines.utils` or a similar shared module. This immediately removes ~40 lines of duplicate code from every pipeline file.

### Recommended Action Items

1.  **Immediate:** Refactor `registry.py` to use a generic class.
2.  **Immediate:** Extract `_ensure_contour` to a common utility.
3.  **Short Term:** Create `EdgeDetectorStrategy` and refactor `edge_detection.py` to remove the `if/elif` chain.
4.  **Medium Term:** Split `MainController` into `CameraManager` and `app_controller`.
