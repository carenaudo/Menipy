# Pint + pint-pydantic Integration Plan

Purpose: Replace `unum` with `Pint` for unit-aware quantities and use `pint-pydantic` for seamless Pydantic integration across GUI, models, and pipelines.

## Objectives
- Standardize units handling on Pint quantities across the codebase.
- Allow users to enter values with units (strings, tuples), validate them, and convert to SI for solvers.
- Minimize churn by keeping existing field names and UX where possible.

## Why Pint
- Mature, widely used scientific units library with dimensional analysis and conversions.
- Great Pydantic support via `pint-pydantic` (Pydantic v2 compatible).
- Cleaner API for `.to("unit").m` to get magnitudes; better ecosystem support than `unum`.

## Dependencies
- Add: `pint`, `pydantic-pint`
- Keep: `pydantic` (v2)
- Remove: `unum` once migration completes

Changes:
- requirements.txt: add `pint`, `pydantic-pint`; eventually drop `unum`.
- pyproject.toml: add to `[project.dependencies]`.

## Registry and Conventions
- Create a shared registry once and reuse everywhere to avoid multiple registries.
- File: `src/menipy/common/units.py`
  - `from pint import UnitRegistry`
  - `ureg = UnitRegistry(auto_reduce_dimensions=True)`
  - Define common aliases: `ureg.define("mN = millinewton")`, `ureg.define("mmH2O = millimeter * water * gravity")` as needed.
  - Expose helpers: `Q_ = ureg.Quantity`

## Pydantic Integration Approach
Two good options (choose A for minimal boilerplate):

A) pydantic-pint type annotations (recommended):
- `from pint import UnitRegistry`
- `from pint.facets.plain import PlainQuantity as Quantity`
- `from pydantic_pint import PintQuantity, UnitRegistrySingleton`
- Configure registry: `UnitRegistrySingleton.set(ureg)` in units module.
- Create type aliases using Annotated constraints, e.g.:
  - `Density = Annotated[Quantity, PintQuantity("kg / m**3")]`
  - `Length = Annotated[Quantity, PintQuantity("mm")]`
  - `SurfaceTension = Annotated[Quantity, PintQuantity("N / m")]`
  - `Angle = Annotated[Quantity, PintQuantity("deg")]`

B) PintModel base class:
- Use `pint-pydantic.PintModel` as `BaseModel` to set a default registry. Still annotate fields with `Quantity` and optionally validators.

We will implement A inside a replacement for `src/menipy/models/unit_types.py` to minimize file-touching elsewhere.

## Data Model Changes
- File: `src/menipy/models/unit_types.py`
  - Replace unum-specific parser/serializer with Pint types and aliases using `PintQuantity` constraints.
  - Provide a permissive parser for inputs (strings like "1000 kg/m^3", tuples `(1000, "kg/m^3")`, and `Quantity`).

- File: `src/menipy/models/config.py`
  - Keep the existing field names. With the new aliases imported from `unit_types.py`, `PhysicsParams` will accept `Quantity` transparently.
  - No other schema changes needed.

## Call Site Updates
- File: `src/menipy/gui/controllers/pipeline_controller.py`
  - Replace `from unum.units import kg, m, mm` with Pint conversions.
  - Convert to SI using Pint:
    - Before: `physics_params.delta_rho.asUnit(kg/m**3).value`
    - After: `physics_params.delta_rho.to("kg/m**3").m`
    - Before: `needle_radius.asUnit(mm).value`
    - After: `physics_params.needle_radius.to("mm").m`
  - Note: `Quantity.to("unit").m` returns the float magnitude.

- Any GUI dialogs that refer to "Unum" in labels or comments should be updated to say "units" or "Pint quantity".

## Step-by-Step Migration
1) Add dependencies
    - requirements.txt: add `pint`, `pydantic-pint` (keep `unum` for now).
    - pyproject.toml: add them to dependencies.

2) Introduce registry and types
   - Add `src/menipy/common/units.py` defining `ureg`, `Q_`, and setting the `UnitRegistrySingleton`.
   - Rewrite `src/menipy/models/unit_types.py` to define `Density`, `Length`, `SurfaceTension`, `Angle` using `PintQuantity`.

3) Update conversions in controller
   - Edit `src/menipy/gui/controllers/pipeline_controller.py:154` and adjacent lines to use `Quantity.to(...).m` instead of `asUnit(...)`.
   - Remove the import of `unum.units`.

4) Validate model parsing
   - Ensure `PhysicsParams` accepts strings like "1000 kg/m^3", "0.5 mm", "72 mN/m". Pint handles prefixes like mN (millinewton) by default.
   - Add a quick test for round-trip serialization of `PhysicsParams` with quantities.

5) Soft deprecate unum
   - Keep `unum` in requirements for one release; add deprecation notes.
   - Provide migration helpers if any external plugins relied on `Unum` types.

6) Remove unum
   - After verifying GUI and pipelines, remove `unum` from requirements and any residual imports.

## Suggested Code Sketches

units module (new):
```
# src/menipy/common/units.py
from pint import UnitRegistry
from pydantic_pint import UnitRegistrySingleton

ureg = UnitRegistry(auto_reduce_dimensions=True)
# Optional aliases
# ureg.define("mN = millinewton")

UnitRegistrySingleton.set(ureg)
Q_ = ureg.Quantity
```

unit types (replace current unum-based):
```
# src/menipy/models/unit_types.py
from typing import Annotated
from pint.facets.plain import PlainQuantity as Quantity
from pydantic_pint import PintQuantity

Density = Annotated[Quantity, PintQuantity("kg / m**3")]
Length = Annotated[Quantity, PintQuantity("mm")]
SurfaceTension = Annotated[Quantity, PintQuantity("N / m")]
Angle = Annotated[Quantity, PintQuantity("deg")]
```

controller conversion (Pint):
```
# src/menipy/gui/controllers/pipeline_controller.py:154
delta_rho_si = physics_params.delta_rho.to("kg/m**3").m if physics_params.delta_rho else 1000.0
needle_diam_mm_si = physics_params.needle_radius.to("mm").m * 2 if physics_params.needle_radius else None
```

## Testing Plan
- Unit tests
  - Parse and validate PhysicsParams with strings, tuples, and Quantity inputs.
  - Verify conversions to SI magnitudes used by pipelines.
  - Ensure serialization produces reasonable strings or base magnitudes as desired.

- GUI smoke tests
  - Enter values with units in Physics dialog; run pendant and sessile analyses; check results.

- Backward compatibility
  - If any data fixtures used unum string formatting, accept equivalent Pint strings (e.g., "kg/m^3").

## Rollout Notes
- Document the change in README and CHANGELOG: "Units now powered by Pint; use strings like '1000 kg/m^3' in settings."
- For plugin authors: accept Pint `Quantity` in configs; convert with `.to(...).m` for floats.

## File Touch List
- Add: `src/menipy/common/units.py`
- Update: `src/menipy/models/unit_types.py`
- Update: `src/menipy/gui/controllers/pipeline_controller.py:13`, `src/menipy/gui/controllers/pipeline_controller.py:154`
- Update docs: references to "Unum" in dialogs and docs.

