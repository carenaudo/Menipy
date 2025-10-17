# src/menipy/models/unit_types.py
"""Custom Pydantic types for handling physical units with the 'unum' library."""

from typing import Any, Annotated

from pydantic import BeforeValidator, PlainSerializer
from unum import Unum
from unum.units import * # Import common units like m, kg, s, etc.


def- unum_parser(v: Any) -> Unum:
    """Parses various inputs into a Unum object."""
    if isinstance(v, Unum):
        return v
    if isinstance(v, str):
        # A simple parser for strings like "1.2 mm" or "72.8 mN/m"
        try:
            return Unum.from_string(v)
        except Exception as e:
            raise ValueError(f"Could not parse '{v}' as a Unum object: {e}")
    if isinstance(v, (int, float)):
        # For now, reject raw numbers to enforce explicit units.
        # In the future, we could allow this with a default unit.
        raise ValueError(f"Raw number '{v}' is not allowed. Please provide a unit (e.g., '{v} mm').")
    if isinstance(v, tuple) and len(v) == 2:
        val, unit_str = v
        try:
            # Create a Unum object from a (value, unit_string) tuple
            return Unum(val, unit_str)
        except Exception as e:
            raise ValueError(f"Could not parse tuple '{v}' as a Unum object: {e}")

    raise TypeError(f"Unsupported type for Unum conversion: {type(v)}")

def unum_serializer(v: Unum) -> str:
    """Serializes a Unum object to a string."""
    return str(v)

# Generic Unum type for Pydantic
UnumType = Annotated[
    Unum,
    BeforeValidator(unum_parser),
    PlainSerializer(unum_serializer, return_type=str),
]

# --- Specialized Unit Types (Recommended) ---

def create_unit_validator(unit: Unum):
    """Factory to create a validator that checks for compatible units."""
    def validator(v: Any) -> Unum:
        unum_obj = unum_parser(v)
        if not unum_obj.is_compatible(unit):
            raise ValueError(f"Value {v} has incompatible units. Expected units compatible with '{unit.str_unit()}'.")
        return unum_obj
    return validator

Length = Annotated[Unum, BeforeValidator(create_unit_validator(m))]
Density = Annotated[Unum, BeforeValidator(create_unit_validator(kg/m**3))]
SurfaceTension = Annotated[Unum, BeforeValidator(create_unit_validator(N/m))]
Angle = Annotated[Unum, BeforeValidator(create_unit_validator(rad))]
