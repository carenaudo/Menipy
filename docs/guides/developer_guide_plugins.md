# Developer Guide: Menipy Plugins

This guide explains the architecture of Menipy plugins and provides a step-by-step walkthrough for creating new ones.

## 1. Core Concepts

Menipy's plugin system allows for extending the application's functionality with new algorithms and processing stages. This is achieved through a system of discovery, registration, and dynamic loading.

### 1.1. Plugin Discovery and Database

Plugins are discovered by scanning designated directories for Python files (`.py`). The core of the plugin management is the `PluginDB` class (`src/menipy/common/plugin_db.py`), which uses a SQLite database (`menipy_plugins.sqlite`) to keep track of all discovered plugins. This database stores metadata about each plugin, such as its name, kind, file path, and whether it is active.

The discovery process is handled by the `discover_into_db` function in `src/menipy/common/plugins.py`. This function scans the plugin directories and updates the database with any new or changed plugins.

### 1.2. Plugin Loading and Registration

Plugins that are marked as "active" in the database are loaded at runtime by the `load_active_plugins` function. This function dynamically imports the plugin's Python module.

Once a module is loaded, it needs to register its functionality. Menipy uses a central `registry` (`src/menipy/common/registry.py`) to keep track of available plugin functions. There are several ways a plugin can register itself:

1.  **Using a `register` function:** The plugin can define a function called `register` that receives a dictionary of registration functions.
2.  **Using predefined dictionaries:** The plugin can define dictionaries like `EDGE_DETECTORS` or `SOLVERS` that map a name to a function.
3.  **Using `get_*` functions:** The plugin can define functions like `get_edge_detectors` or `get_solvers` that return a dictionary of plugins.

### 1.3. Plugin Types (Kinds)

Each plugin has a "kind" that determines what part of the pipeline it extends. The kind is typically inferred from the filename. For example, a file named `my_edge_detector.py` would be identified as an "edge" plugin. Some of the common kinds include:

*   `acquisition`
*   `preprocessing`
*   `edge_detection`
*   `geometry`
*   `scaling`
*   `physics`
*   `solver`
*   `optimization`
*   `outputs`
*   `overlay`
*   `validation`

## 2. Creating a New Plugin

Here’s a step-by-step guide to creating a new plugin. We'll create a simple "edge detection" plugin.

### Step 1: Create the Plugin File

Create a new Python file in one of the plugin directories (e.g., the `plugins/` directory at the root of the project). The name of the file is important, as it helps determine the plugin's name and kind. Let's call our file `my_edge_detector.py`.

### Step 2: Write the Plugin Function

Inside `my_edge_detector.py`, write the function that implements the plugin's logic. For an edge detection plugin, this function will typically take an image as input and return a set of coordinates representing the detected edge.

```python
# plugins/my_edge_detector.py

import numpy as np

def my_detector(image):
    """A simple edge detector that returns a circular contour."""
    h, w = image.shape[:2]
    center_x, center_y = w // 2, h // 2
    radius = min(h, w) * 0.4
    t = np.linspace(0, 2 * np.pi, 100)
    x = center_x + radius * np.cos(t)
    y = center_y + radius * np.sin(t)
    return np.column_stack([x, y])
```

#### Debug Mode Support (Optional)

Detectors can optionally support a "Debug Mode" to return intermediate candidates or scoring info. To support this:

1.  Accept a `return_debug: bool = False` argument in your detect method.
2.  If `True`, return a tuple: `(result_contour, debug_info_list)`.
3.  `debug_info_list` should be a list of tuples: `(contour, score, label)`.

```python
def my_debuggable_detector(image, settings, return_debug=False):
    # ... find candidates ...
    best_contour = ...
    candidates = [(cnt1, 0.8, "candidate A"), (cnt2, 0.5, "candidate B")]
    
    if return_debug:
        return best_contour, candidates
    return best_contour
```
```

### Step 3: Register the Plugin

Now, you need to register your plugin so that Menipy can find it. As mentioned earlier, there are a few ways to do this. Here are three examples:

**Option A: Using a `register` function**

This is the most explicit way to register a plugin.

```python
# ... (your function code) ...

from menipy.common.registry import register_edge

def register(registries):
    registries['register_edge']('my-circle', my_detector)
```

**Option B: Using a predefined dictionary**

This is a simpler way to register a plugin if you are providing a single function for a specific category.

```python
# ... (your function code) ...

EDGE_DETECTORS = {
    'my-circle': my_detector
}
```

**Option C: Using a `get_*` function**

This is useful if you want to dynamically generate your plugin functions.

```python
# ... (your function code) ...

def get_edge_detectors():
    return {
        'my-circle': my_detector
    }
```

### Step 4: Activate the Plugin

After creating the plugin, you need to make sure it's discovered and activated. The application will typically discover plugins on startup. You can then use the GUI or a command-line tool to activate or deactivate your plugin.

## 3. Example Plugins

The `plugins/` directory in the Menipy project contains several example plugins that you can use as a reference. Some good examples to look at are:

*   `acq_file.py`: A simple acquisition plugin.
*   `bezier_edge.py`: An edge detection plugin.
*   `physics_dummy.py`: A dummy physics plugin.
*   `simple_solver.py`: A basic solver plugin.
