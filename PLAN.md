# Menipy Implementation Plan

This document outlines the current development roadmap for Menipy. It is a living document that will be updated as features are implemented.

## Phase 1: Core Scientific Model Enhancement

The immediate priority is to transition from direct estimation methods to a full, optimization-based scientific model, which is the standard for accurate tensiometry.

- **1.1: Full ADSA Integration**: The main task is to develop and integrate an iterative optimization loop that fits the theoretical Young-Laplace profile to the detected droplet contour. This will provide highly accurate surface tension and contact angle measurements.

- **1.2: GUI Unit Management**: To improve user experience and flexibility, the GUI will be updated to handle physical units. Users will be able to input parameters in various units (e.g., g/cm³, mN/m), and the backend will perform all calculations in base SI units.

## Phase 2: Expansion of Analysis Capabilities

Once the core ADSA model is robust, the project will expand to include other common tensiometry methods.

- **2.1: New Analysis Pipelines**:
  - **Captive Bubble**: Develop the full analysis pipeline for captive bubble measurements, which is critical for analyzing surfactants and complex interfaces.
  - **Capillary Rise**: Implement the workflow for analyzing capillary rise, providing another method for surface tension measurement.

- **2.2: Additional Geometric Models**:
  - Implement supplementary models for sessile drop analysis, such as ellipse fitting, to provide more tools for geometric characterization.

## Phase 3: Documentation and Maintenance

As development proceeds, documentation will be kept in sync with the implementation to ensure the project remains maintainable and accessible to new contributors.

- **3.1: Documentation Synchronization**: Systematically review and update all foundational documents in the `docs/guides/` directory to reflect the latest implemented features and code architecture.

## 4. Pipeline Specific Plans

Detailed implementation plans for each pipeline can be found in the following documents:

- [Pendant Drop Pipeline](file:///d:/programacion/Menipy/pendant_plan_pipeline.md)
- [Sessile Drop Pipeline](file:///d:/programacion/Menipy/sessile_plan_pipeline.md)
- [Capillary Rise Pipeline](file:///d:/programacion/Menipy/capillary_rise_plan_pipeline.md)
- [Oscillating Drop Pipeline](file:///d:/programacion/Menipy/oscillating_plan_pipeline.md)
- [Captive Bubble Pipeline](file:///d:/programacion/Menipy/captive_bubble_plan_pipeline.md)
