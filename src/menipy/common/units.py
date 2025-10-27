# src/menipy/common/units.py
from pint import UnitRegistry
from pydantic_pint import set_registry

ureg = UnitRegistry(auto_reduce_dimensions=True)
# Optional aliases
# ureg.define("mN = millinewton")

set_registry(ureg)
Q_ = ureg.Quantity