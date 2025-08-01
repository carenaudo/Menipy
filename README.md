# Menipy

Menipy is a Python-based toolkit aimed at analyzing droplet shapes from images. The goal is to create the software with minimal human involvement in the coding phase. Development is driven by CODEX, an AI agent that orchestrates several specialized sub-agents defined in `AGENTS.md`.

These sub-agents read the step-by-step instructions in `PLAN.md` and consult the reference material under `doc/` to automatically scaffold the project, implement processing and modeling algorithms, build the PySide6 GUI, and configure testing and packaging.

## Repository Overview

- **AGENTS.md** – roles for Documentation, Scaffold, Environment, Processing, Modeling, GUI, Batch, and CI & Packaging agents.
- **PLAN.md** – a high-level plan detailing the desired directory layout, technology stack, and feature set.
- **doc/** – supporting Markdown files (`physics_models.md`, `numerical_methods.md`, `image_processing.md`, `gui_design.md`, `drop_analysis.md`) that supply equations and workflow descriptions for CODEX.

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
