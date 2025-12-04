# src/menipy/models/unit_types.py
"""
Pydantic-Pint unit types for physical quantities.
"""
from typing import Annotated
from pint.facets.plain import PlainQuantity as Quantity
from pydantic_pint import PydanticPintQuantity

Density = Annotated[Quantity, PydanticPintQuantity("kg / m**3")]
Length = Annotated[Quantity, PydanticPintQuantity("mm")]
SurfaceTension = Annotated[Quantity, PydanticPintQuantity("N / m")]
Angle = Annotated[Quantity, PydanticPintQuantity("deg")]
