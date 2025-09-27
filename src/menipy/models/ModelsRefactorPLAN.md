# Models Refactoring Plan for Menipy

## 1. Objective

The primary goal of this refactoring is to reorganize and standardize the data type definitions within the `src/menipy/models/` directory. This will improve:

-   **Readability**: Models will be grouped logically by their role in the pipeline.
-   **Maintainability**: Easier to locate and modify specific data structures.
-   **Alignment with Pipeline Design**: The structure will mirror the flow of data through the Menipy analysis pipeline.
-   **Consistency**: Standardize on Pydantic `BaseModel` for all data structures, leveraging its validation and serialization features.

## 2. Proposed Directory Structure

The `src/menipy/models/` directory will be restructured as follows:

```
src/menipy/models/
├── __init__.py
├── context.py         # The main Context object that carries data through the pipeline.
├── config.py          # User-configurable parameters (optical, physical, solver settings).
├── frame.py           # Models for raw and processed image data (CameraMeta, Calibration, Frame).
├── geometry.py        # Models for geometric constructs (Contour, Geometry, Point, ROI, etc.).
├── fit.py             # Models for numerical fitting results and configuration (SolverInfo, Residuals, Confidence, FitConfig).
└── result.py          # Models for final, high-level analysis outputs (YoungLaplaceFit, OscillationFit, CapillaryRiseFit).
```

## 3. Refactoring Phases

### Phase 0: Preparation & Baseline

1.  **Review Existing Models**:
    *   Identify all current data classes in `src/menipy/models/datatypes.py` and `src/menipy/models/params.py`.
    *   Note their dependencies and where they are used throughout the codebase.
2.  **Ensure Test Coverage**:
    *   Verify that existing unit and integration tests adequately cover the functionality relying on these models. This provides a safety net for the refactoring.
3.  **Create New Directory Structure**:
    *   Create the new empty `.py` files (`context.py`, `config.py`, `frame.py`, `geometry.py`) within `src/menipy/models/`.
    *   (`fit.py` and `result.py` have already been created and partially populated.)

### Phase 1: Core Model Migration & Standardization

This phase involves moving existing data classes to their new, dedicated files and converting them to Pydantic `BaseModel` where necessary.

1.  **Populate `fit.py` and `result.py`**: (Already completed in previous steps)
    *   `SolverInfo`, `Residuals`, `Confidence`, `FitConfig` are in `fit.py`.
    *   `YoungLaplaceFit`, `OscillationFit`, `CapillaryRiseFit` are in `result.py`.

2.  **Populate `config.py`**:
    *   Move `PhysicsParams` from `datatypes.py` to `config.py`.
    *   Move all preprocessing and edge detection settings (`ResizeSettings`, `FilterSettings`, `BackgroundSettings`, `NormalizationSettings`, `ContactLineSettings`, `PreprocessingSettings`, `EdgeDetectionSettings`) from `datatypes.py` to `config.py`.
    *   Ensure all moved classes are Pydantic `BaseModel`s.

3.  **Populate `frame.py`**:
    *   Move `CameraMeta`, `Calibration`, `Frame` from `datatypes.py` to `frame.py`.
    *   Ensure all moved classes are Pydantic `BaseModel`s.

4.  **Populate `geometry.py`**:
    *   Move `Contour`, `Geometry` from `datatypes.py` to `geometry.py`.
    *   Move `Point`, `ROI`, `Needle`, `ContactLine` from the existing `src/menipy/models/geometry.py` (which currently contains utility functions) into this new `models/geometry.py` file, converting them to Pydantic `BaseModel`s. The utility functions in the original `geometry.py` should remain there or be moved to a `src/menipy/common/geometry.py` if they are general-purpose.

5.  **Define `context.py`**:
    *   Create the `Context` class in `context.py`. This class will be the central data carrier for the pipeline.
    *   Its attributes should be instances of the newly organized Pydantic models (e.g., `config: Config`, `current_frame: Frame`, `detected_geometry: Geometry`, `fit_results: Fit`, `final_output: Result`).
    *   Replace the current `dataclass Context` with this new Pydantic-based `Context`.

### Phase 2: Updating References & Integration

1.  **Update Import Statements**:
    *   Go through the entire codebase and update all `import` statements to reflect the new locations of the models (e.g., `from .datatypes import Frame` becomes `from .frame import Frame`).
2.  **Refactor `Context` Usage**:
    *   Modify any code that directly accesses `Context` attributes using dictionary-like access (e.g., `context.physics['delta_rho']`) to use the new Pydantic-based attribute access (e.g., `context.config.physics.delta_rho`). This is a critical step that will touch many pipeline stages.
3.  **Review `AnalysisRecord`**:
    *   The `AnalysisRecord` class in `datatypes.py` aggregates various results. Decide if it should remain in `datatypes.py` (as a general record) or be moved to `result.py` (as a high-level output).
    *   Ensure `AnalysisRecord` uses the new, refactored models for its attributes (e.g., `calibration: Calibration`, `physics: PhysicsParams`, `fit_young_laplace: YoungLaplaceFit`).

### Phase 3: Cleanup & Validation

1.  **Clean Up `datatypes.py` and `params.py`**:
    *   Once all classes have been successfully moved and their references updated, remove the redundant definitions from `datatypes.py` and `params.py`.
    *   `datatypes.py` might remain for general type aliases (like `ImageGray`, `FloatVec`, `ContourArray`) if they are still widely used.
2.  **Run All Tests**:
    *   Execute the full test suite to ensure that no regressions have been introduced and that the application functions correctly with the new model structure.
3.  **Update Documentation**:
    *   Update `NEW_PLAN.md`, `README.md`, and any other relevant documentation to reflect the new model organization and the consistent use of Pydantic.

## 4. Key Considerations

-   **Pydantic Consistency**: All data classes should ultimately be Pydantic `BaseModel`s. This provides automatic data validation, clear schema definition, and easy serialization/deserialization.
-   **Type Hinting**: Maintain and improve type hinting throughout the models and any functions interacting with them.
-   **Incremental Changes**: Perform the refactoring in small, testable steps to minimize disruption and simplify debugging.
-   **Backward Compatibility**: This refactoring will intentionally break direct attribute access to the `Context` object and the location of many data models. This is a necessary trade-off for a more robust and maintainable architecture.

This plan provides a structured approach to achieve a cleaner, more maintainable, and pipeline-aligned `models` directory.
```