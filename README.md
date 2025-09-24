# Menipy

The current main branch is under current refactorization of file structure. **The main branch is currently NOT WORKINg.**
To test the original Menipy code made with minimal interaction use the branch: **https://github.com/carenaudo/Menipy/tree/Min-Human-Interaction**


Menipy is a Python-based toolkit aimed at analyzing droplet shapes from images. The goal is to create the software with minimal human involvement in the coding phase. Development is driven by CODEX, an AI agent that orchestrates several specialized sub-agents defined in `AGENTS.md`.

These sub-agents read the step-by-step instructions in `PLAN.md` and consult the reference material under `doc/` to automatically scaffold the project, implement processing and modeling algorithms, build the PySide6 GUI, and configure testing and packaging.

<img width="1909" height="1028" alt="image" src="https://github.com/user-attachments/assets/45fd8f53-98cc-4c6d-bf59-0e0687c5c7fe" />

# ​​ Disclaimer

**This software is under development and is *not production-ready***. It is *not intended to replace any commercial or non-commercial tools or software*. The values reported by this software **may be incorrect or inaccurate**, and **the developers assume no responsibility** whatsoever for how the software is used or for any consequences arising from its use.

Use at your own risk.

## Repository Overview

- **AGENTS.md** – roles for Documentation, Scaffold, Environment, Processing, Modeling, GUI, Batch, and CI & Packaging agents.
- **PLAN.md** – a high-level plan detailing the desired directory layout, technology stack, and feature set.
- **doc/** – supporting Markdown files (`physics_models.md`, `numerical_methods.md`, `image_processing.md`, `gui_design.md`, `drop_analysis.md`) that supply equations and workflow descriptions for CODEX.
- **CHANGELOG.md** – A log of notable changes for each version.

By combining these materials with CODEX automation, Menipy aims to become a fully functional droplet shape analysis tool with a PySide6 interface and automated tests, all generated with minimal human interaction.

## Human Interaction

Human involvement in this project is intentionally kept light. The main role of
the user will be testing the GUI and submitting prompts as tasks to CODEX or as
issues on GitHub. Feedback from these manual tests will guide further automated
iterations of the tool.

## Calibration Workflow

1. Enable **Calibration Mode** in the parameter panel.
2. Toggle **Manual Calibration** on to draw a line between two points, or off to
   draw the needle region for automatic detection.
3. Click **Calibrate** to compute the pixel-to-mm scale. The resulting value is
   shown in the parameter panel.
   Automatic mode detects the two vertical edges of the needle using Canny edge
   detection and a Hough transform to measure their separation.

## Image Processing

Segmented masks are cleaned with morphological operations and reduced to the largest external contour. This ensures internal artifacts do not affect volume or fitting calculations.

## Drop Analysis

The GUI includes a **Drop Analysis** tab for pendant and contact-angle modes. After defining
needle and drop regions, the application detects the needle width, extracts the
outer drop contour and overlays key features such as the apex and symmetry axis.
Metrics including height, diameter, volume and surface tension are displayed in
the panel. Detailed instructions are available in `doc/drop_analysis.md`.

## Developer note: runtime resources and GUI logging

During recent refactor work, the application now emits clear per-stage pipeline logs and a status message that is shown in the GUI status bar. If you run the GUI and see missing icon warnings for paths like `:/icons/...`, run the resource build helper:

- In the project root, run the build script to produce `icons.rcc` (requires PySide6 tools):
  - Windows PowerShell:
    ```powershell
    D:/programacion/Menipy/.venv/Scripts/python.exe tools/build_resources.py
    ```

After producing `icons.rcc`, the application will register it automatically at startup and QIcon lookups for `:/icons/...` should resolve. If you prefer a quick workaround, the GUI will fall back to the on-disk SVG files under `src/menipy/gui/resources/icons/` when the compiled resource is not available.

Logs: the application forwards Python logging to the in-app Log tab. Look there for per-stage messages like:
```
[pipeline:<name> ctx=0x...] Completed stage: <stage> (<ms> ms) | frames=N preview=M contour=K results_keys=L
```

## Developer Documentation

For developers looking to extend Menipy's functionality, we provide guides on common tasks.
- 
- [**Adding a New Analysis Pipeline**](doc/developer_guide_pipelines.md): A step-by-step guide on how to create and integrate a new analysis workflow into the application.
- [**Adding a New Image Filter Plugin**](doc/developer_guide_plugins.md): A guide on creating and integrating custom image filters using the plugin system.
```markdown
# Menipy

The current main branch is under current refactorization of file structure. **The main branch is currently NOT WORKINg.**
To test the original Menipy code made with minimal interaction use the branch: **https://github.com/carenaudo/Menipy/tree/Min-Human-Interaction**


Menipy is a Python-based toolkit aimed at analyzing droplet shapes from images. The goal is to create the software with minimal human involvement in the coding phase. Development is driven by CODEX, an AI agent that orchestrates several specialized sub-agents defined in `AGENTS.md`.

These sub-agents read the step-by-step instructions in `PLAN.md` and consult the reference material under `doc/` to automatically scaffold the project, implement processing and modeling algorithms, build the PySide6 GUI, and configure testing and packaging.

<img width="1909" height="1028" alt="image" src="https://github.com/user-attachments/assets/45fd8f53-98cc-4c6d-bf59-0e0687c5c7fe" />

# Disclaimer

**This software is under development and is *not production-ready***. It is *not intended to replace any commercial or non-commercial tools or software*. The values reported by this software **may be incorrect or inaccurate**, and **the developers assume no responsibility** whatsoever for how the software is used or for any consequences arising from its use.

Use at your own risk.

## Repository Overview

- **AGENTS.md** – roles for Documentation, Scaffold, Environment, Processing, Modeling, GUI, Batch, and CI & Packaging agents.
- **PLAN.md** – a high-level plan detailing the desired directory layout, technology stack, and feature set.
- **doc/** – supporting Markdown files (`physics_models.md`, `numerical_methods.md`, `image_processing.md`, `gui_design.md`, `drop_analysis.md`) that supply equations and workflow descriptions for CODEX.
- **CHANGELOG.md** – A log of notable changes for each version.

By combining these materials with CODEX automation, Menipy aims to become a fully functional droplet shape analysis tool with a PySide6 interface and automated tests, all generated with minimal human interaction.

## Human Interaction

Human involvement in this project is intentionally kept light. The main role of
the user will be testing the GUI and submitting prompts as tasks to CODEX or as
issues on GitHub. Feedback from these manual tests will guide further automated
iterations of the tool.

## Calibration Workflow

1. Enable **Calibration Mode** in the parameter panel.
2. Toggle **Manual Calibration** on to draw a line between two points, or off to
   draw the needle region for automatic detection.
3. Click **Calibrate** to compute the pixel-to-mm scale. The resulting value is
   shown in the parameter panel.
   Automatic mode detects the two vertical edges of the needle using Canny edge
   detection and a Hough transform to measure their separation.

## Image Processing

Segmented masks are cleaned with morphological operations and reduced to the largest external contour. This ensures internal artifacts do not affect volume or fitting calculations.

## Drop Analysis

The GUI includes a **Drop Analysis** tab for pendant and contact-angle modes. After defining
needle and drop regions, the application detects the needle width, extracts the
outer drop contour and overlays key features such as the apex and symmetry axis.
Metrics including height, diameter, volume and surface tension are displayed in
the panel. Detailed instructions are available in `doc/drop_analysis.md`.

## Developer note: runtime resources and GUI logging

During recent refactor work, the application now emits clear per-stage pipeline logs and a status message that is shown in the GUI status bar. If you run the GUI and see missing icon warnings for paths like `:/icons/...`, run the resource build helper:

- In the project root, run the build script to produce `icons.rcc` (requires PySide6 tools):
  - Windows PowerShell:
    ```powershell
    D:/programacion/Menipy/.venv/Scripts/python.exe tools/build_resources.py
    ```

After producing `icons.rcc`, the application will register it automatically at startup and QIcon lookups for `:/icons/...` should resolve. If you prefer a quick workaround, the GUI will fall back to the on-disk SVG files under `src/menipy/gui/resources/icons/` when the compiled resource is not available.

Logs: the application forwards Python logging to the in-app Log tab. Look there for per-stage messages like:
```
[pipeline:<name> ctx=0x...] Completed stage: <stage> (<ms> ms) | frames=N preview=M contour=K results_keys=L
```

## Developer Documentation

For developers looking to extend Menipy's functionality, we provide guides on common tasks.

- [**Adding a New Analysis Pipeline**](doc/developer_guide_pipelines.md): A step-by-step guide on how to create and integrate a new analysis workflow into the application.
- [**Adding a New Image Filter Plugin**](doc/developer_guide_plugins.md): A guide on creating and integrating custom image filters using the plugin system.

```
