# Menipy

Menipy is a Python-based toolkit aimed at analyzing droplet shapes from images. The goal is to create the software with minimal human involvement in the coding phase. Development is driven by CODEX, an AI agent that orchestrates several specialized sub-agents defined in `AGENTS.md`.

These sub-agents read the step-by-step instructions in `PLAN.md` and consult the reference material under `doc/` to automatically scaffold the project, implement processing and modeling algorithms, build the PySide6 GUI, and configure testing and packaging.

## Repository Overview

- **AGENTS.md** – roles for Documentation, Scaffold, Environment, Processing, Modeling, GUI, Batch, and CI & Packaging agents.
- **PLAN.md** – a high-level plan detailing the desired directory layout, technology stack, and feature set.
- **doc/** – supporting Markdown files (`physics_models.md`, `numerical_methods.md`, `image_processing.md`, `gui_design.md`) that supply equations and workflow descriptions for CODEX.

By combining these materials with CODEX automation, Menipy aims to become a fully functional droplet shape analysis tool with a PySide6 interface and automated tests, all generated with minimal human interaction.
