# src/menipy/models/unit_types.py
"""Pydantic-Pint unit types for physical quantities."""

from typing import Annotated

from pint.facets.plain import PlainQuantity as Quantity
from pydantic_pint import PydanticPintQuantity

from menipy.common.units import ureg

Density = Annotated[Quantity, PydanticPintQuantity("kg / m**3", ureg=ureg)]
Length = Annotated[Quantity, PydanticPintQuantity("mm", ureg=ureg)]
SurfaceTension = Annotated[Quantity, PydanticPintQuantity("N / m", ureg=ureg)]
Angle = Annotated[Quantity, PydanticPintQuantity("deg", ureg=ureg)]
