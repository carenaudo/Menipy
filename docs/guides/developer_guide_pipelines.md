# Developer Guide: Menipy Pipelines

This guide explains the architecture of Menipy pipelines and provides a step-by-step walkthrough for creating new ones.

## 1. Core Concepts

Menipy's analysis capabilities are built around a flexible pipeline architecture. This design allows for modular, reusable, and easily extensible data processing workflows.

### 1.1. `PipelineBase`: The Skeleton

The foundation of all pipelines is the `PipelineBase` class (`src/menipy/pipelines/base.py`). It defines a series of stages that a data processing workflow typically follows. The current canonical stage names are:

1.  **`acquisition`**: Acquire raw data, such as loading an image from a file.
2.  **`preprocessing`**: Filter, crop, normalize, and clean the data.
3.  **`feature_detection`**: Detect supporting features such as ROI, needle, substrate, and contact points.
4.  **`contour_extraction`**: Extract the object contour from the preprocessed image.
5.  **`contour_refinement`**: Clip, smooth, or otherwise clean the contour.
6.  **`calibration`**: Convert pixel measurements to physical units.
7.  **`geometric_features`**: Extract geometric features from the contour.
8.  **`physics`**: Define physical model parameters such as densities and gravity.
9.  **`profile_fitting`**: Fit the physical/model profile to the measured contour.
10. **`compute_metrics`**: Collect final numerical results.
11. **`overlay`**: Generate visualizations for the preview panel.
12. **`validation`**: Perform quality checks on the results.

The `PipelineBase` class provides a `run` method that executes these stages in a predefined order. Each stage is implemented as a `do_*` method (e.g., `do_acquisition`, `do_preprocessing`).

Legacy stage method aliases such as `do_edge_detection`, `do_geometry`,
`do_scaling`, `do_solver`, and `do_outputs` still exist for compatibility, but
new pipelines should use the canonical names above.

### 1.2. The `Context` Object

A central piece of the pipeline is the `Context` object. This is a simple data container that is passed from one stage to the next. Each stage can read data from the `Context`, perform its processing, and then write its results back to the `Context`. This allows for a loosely coupled architecture where stages only need to know about the `Context`, not about each other.

### 1.3. Stage Testing in the GUI

The GUI exposes a Science-mode **Step Test** panel for inspecting pipeline stage
behavior. It is available from **View -> Focus -> Science**, then the **Test**
button in the workflow bar.

For developers, this panel is useful because it exercises the same pipeline
stage plan used by normal execution:

```python
pipeline.run_with_plan(only=[stage_name], include_prereqs=True, **run_kwargs)
```

The panel intentionally excludes `acquisition` from the selectable stage list.
Instead, it uses the selected GUI source and silently runs `AutoCalibrator` to
provide prerequisite context fields such as `roi`, `needle_rect`,
`substrate_line`, `drop_contour`, `contact_points`, and `apex_point`. If a
feature cannot be detected, the panel reports an inline prerequisite warning.

Stage configuration edits in the panel are sandboxed. The controller clones live
preprocessing, edge-detection, geometry, physics, and overlay settings for test
runs. **Apply** writes sandbox values back to the live controllers/settings;
**Discard** reloads from the live state.

### 1.4. Pipeline Discovery

Pipelines are discovered automatically by the `discover.py` module (`src/menipy/pipelines/discover.py`). To be discovered, a pipeline must meet the following criteria:

1.  It must be located in a subdirectory within the `src/menipy/pipelines` directory.
2.  The subdirectory's `__init__.py` file must import a class that inherits from `PipelineBase`.

The name of the subdirectory becomes the identifier for the pipeline.

## 2. Creating a New Pipeline

Here’s a step-by-step guide to creating a new pipeline. As an example, we'll imagine creating a new pipeline called `my_pipeline`.

### Step 1: Create the Directory Structure

First, create a new subdirectory in `src/menipy/pipelines`:

```
src/
└── menipy/
    └── pipelines/
        ├── pendant/
        ├── sessile/
        └── my_pipeline/   <-- New pipeline directory
```

Inside the `my_pipeline` directory, create two files: `__init__.py` and `stages.py`.

```
my_pipeline/
├── __init__.py
└── stages.py
```

### Step 2: Define the Pipeline Class

In `stages.py`, define your new pipeline class. It must inherit from `PipelineBase`.

```python
# src/menipy/pipelines/my_pipeline/stages.py

from menipy.pipelines.base import PipelineBase
from menipy.models.datatypes import Context
from typing import Optional

class MyPipeline(PipelineBase):
    """A brief description of what your pipeline does."""

    name = "my_pipeline"

    def do_acquisition(self, ctx: Context) -> Optional[Context]:
        # Your acquisition logic here
        print("Running acquisition stage for my_pipeline")
        return ctx

    def do_preprocessing(self, ctx: Context) -> Optional[Context]:
        # Your preprocessing logic here
        print("Running preprocessing stage for my_pipeline")
        return ctx

    # ... implement other do_* methods as needed ...
```

You only need to override the stages that are relevant to your pipeline. The base class provides default implementations that do nothing.

### Step 3: Expose the Pipeline

In `__init__.py`, import and expose your new pipeline class.

```python
# src/menipy/pipelines/my_pipeline/__init__.py

from .stages import MyPipeline

__all__ = ["MyPipeline"]
```

### Step 4: Use Common Utilities (Optional)

Menipy provides a set of common utility functions in `src/menipy/common` for tasks like edge detection, solving, and creating overlays. You are encouraged to use these to avoid code duplication.

For example, to run Canny edge detection in your `do_contour_extraction` stage:

```python
from menipy.common import edge_detection as edged

def do_contour_extraction(self, ctx: Context) -> Optional[Context]:
    edged.run(ctx, method="canny")
    return ctx
```

## 3. Examples: `pendant` and `sessile` Pipelines

The existing `pendant` and `sessile` pipelines are excellent examples to learn from. You can find their implementations in:

*   `src/menipy/pipelines/pendant/`
*   `src/menipy/pipelines/sessile/`

By examining their `stages.py` files, you can see how they implement the different pipeline stages and how they use the common utility modules.
