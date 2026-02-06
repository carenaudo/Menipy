# src/menipy/common/units.py
"""Unit registry configuration using Pint."""
from pint import UnitRegistry
from pydantic_pint import set_registry

ureg: UnitRegistry = UnitRegistry(auto_reduce_dimensions=True)
# Optional aliases
# ureg.define("mN = millinewton")

set_registry(ureg)
Q_ = ureg.Quantity
