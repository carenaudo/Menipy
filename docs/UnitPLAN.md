# Plan for GUI-Side Unit Management with `unum`

This document outlines the detailed steps to integrate the `unum` package into the Menipy application. The core architectural principle is to confine unit-aware objects (`Unum` instances) to the user-facing GUI and configuration layers, while keeping the backend analysis and numerical computation pipelines strictly in base SI units using standard floats.

This separation of concerns provides user flexibility without adding complexity or performance overhead to the core scientific code.

---

## Phase 1: Create a Custom Pydantic Type for `Unum`

The first step is to teach Pydantic how to understand, validate, and serialize `Unum` objects. This will be done by creating a reusable, custom Pydantic type.

1.  **Create a New Module:**
    -   Create a new file: `src/menipy/models/unit_types.py`.
    -   This module will house all custom types related to physical units.

2.  **Implement a Generic `UnumType`:**
    -   In `unit_types.py`, define a generic `UnumType` using `typing.Annotated`.
    -   **Parser (`BeforeValidator`):** Create a validation function that can parse various inputs into a `Unum` object:
        -   An existing `Unum` object.
        -   A string with a value and unit (e.g., `"72.8 mN/m"`). This will require a simple string parser.
        -   A tuple `(value, unit_str)`, e.g., `(1.2, "mm")`.
        -   A raw number (`int` or `float`), which should either be rejected or assigned a default unit if context allows (initially, we will reject it to enforce explicit units).
    -   **Serializer (`PlainSerializer`):** Create a serialization function that converts a `Unum` object back into a clean string representation (e.g., `"72.8 mN/m"`). This will be used for saving configurations and displaying values.

3.  **Implement Specialized Unit Types (Optional but Recommended):**
    -   In the same module, create specialized types that inherit from the generic `UnumType` but add unit-consistency checks.
    -   Examples:
        -   `Length = Annotated[Unum, BeforeValidator(lambda v: parse_and_check(v, u.m))]`
        -   `Density = Annotated[Unum, BeforeValidator(lambda v: parse_and_check(v, u.kg/u.m**3))]`
        -   `SurfaceTension = Annotated[Unum, BeforeValidator(lambda v: parse_and_check(v, u.N/u.m))]`
    -   These types will ensure that a field expecting a length cannot be given a mass, for example.

---

## Phase 2: Update Pydantic Models for GUI Configuration

With the custom types defined, we can now update the Pydantic models that are used to capture user settings from the GUI.

1.  **Identify Target Models:**
    -   The primary target is `PhysicsParams` in `src/menipy/models/datatypes.py`. This model holds physical constants and guesses provided by the user.

2.  **Refactor Model Fields:**
    -   Modify the fields in `PhysicsParams` to use the new custom unit types.
    -   **Example (Before):**
        ```python
        class PhysicsParams(BaseModel):
            delta_rho: Optional[float] = Field(default=None, description="density difference Δρ [kg/m^3]")
            needle_radius_mm: Optional[float] = Field(default=None, ge=0)
        ```
    -   **Example (After):**
        ```python
        from .unit_types import Density, Length

        class PhysicsParams(BaseModel):
            delta_rho: Optional[Density] = Field(default=None, description="Density difference Δρ")
            needle_radius: Optional[Length] = Field(default=None)
        ```
    -   Update the field descriptions to remove hardcoded units, as the user can now provide them in any compatible format.

---

## Phase 3: Adapt the GUI Layer

This phase involves modifying the `MainWindow` and its associated panels to handle the flow of `Unum` objects.

1.  **Update Input Panels:**
    -   In the GUI panels (e.g., `AnalysisTab`), the `QLineEdit` widgets for physical parameters will now accept strings with units (e.g., `"0.998 g/cm**3"`).
    -   When the "Analyze" button is clicked, the GUI controller will gather these strings and use them to instantiate the `PhysicsParams` model. Pydantic will automatically handle the parsing and validation.

2.  **Implement Conversion to SI Units:**
    -   In `MainWindow._run_analysis` (or a similar controller method), before calling the backend analysis functions, convert all `Unum` values from the `PhysicsParams` object into raw SI floats.
    -   **Example:**
        ```python
        # Get Unum object from the config model
        delta_rho_unum = physics_params.delta_rho

        # Convert to SI float for the backend
        si_params = {
            "delta_rho": delta_rho_unum.asUnit(u.kg / u.m**3).value,
            # ... other parameters
        }

        # Call backend with pure floats
        metrics = compute_drop_metrics(..., physics=si_params)
        ```

3.  **Update Results Display:**
    -   The backend will return a dictionary of results as SI floats.
    -   The GUI controller will take these floats, re-attach the base SI units, and store them as `Unum` objects.
    -   **Example:** `gamma_unum = metrics["gamma_mN_m"] * (u.N / u.m)`
    -   The `AnalysisTab.set_metrics` method will be updated to accept these `Unum` objects. It will then format them for display, potentially converting them to more conventional units (e.g., `mN/m`).
    -   **Example:** `self.gamma_label.setText(f"{gamma_unum.asUnit(u.mN/u.m):.4f}")`

---

## Phase 4: Refactor Backend and Finalize

The final phase is to ensure the backend is completely decoupled from any unit library.

1.  **Verify Backend Function Signatures:**
    -   Audit all functions in `menipy.analysis`, `menipy.pipelines`, and `menipy.processing`.
    -   Confirm that all function arguments and return values related to physical quantities are typed as `float` or `np.ndarray`.

2.  **Update Docstrings:**
    -   This is a critical step for maintainability. Update the docstring for every function parameter to explicitly state the **required SI unit**.
    -   **Example:**
        ```python
        def surface_tension(delta_rho: float, r0: float) -> float:
            """Calculates surface tension.

            Args:
                delta_rho (float): Density difference in kg/m³.
                r0 (float): Apex radius in meters.
            """
        ```

3.  **Remove Obsolete Fields:**
    -   Remove fields with hardcoded units like `needle_radius_mm` from `PhysicsParams` in favor of the new `Unum`-based fields.