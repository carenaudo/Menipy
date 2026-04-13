# src/menipy/common/units.py
"""Unit registry configuration using Pint."""
from pint import UnitRegistry
from pydantic_pint import set_registry

ureg: UnitRegistry = UnitRegistry(auto_reduce_dimensions=True)
# Optional aliases
# ureg.define("mN = millinewton")

set_registry(ureg)
Q_ = ureg.Quantity

# Conversion Factors
# Internal SI (kg, m, s, N) <-> User Display (SI vs CGS)
# Standard units are typically:
# Density: kg/m^3 (SI) vs g/cm^3 (CGS) -> 1000
# Surface Tension: mN/m (SI) vs dyn/cm (CGS) -> 1 (identical)
# Length: mm (SI) vs cm (CGS) -> 10

def convert_to_si(value: float, quantity: str, source_system: str) -> float:
    """Convert display value to internal SI."""
    if source_system == "SI":
        return value
    
    if quantity == "density":
        return value * 1000.0  # g/cm^3 -> kg/m^3
    if quantity == "length":
        return value * 10.0    # cm -> mm (wait, internal is mm or m?)
    # Most internal math uses mm for coordinates, but kg/m^3 for density
    
    return value

def convert_from_si(value: float, quantity: str, target_system: str) -> float:
    """Convert internal SI to display system."""
    if target_system == "SI":
        return value
        
    if quantity == "density":
        return value / 1000.0  # kg/m^3 -> g/cm^3
    if quantity == "length":
        return value / 10.0    # mm -> cm
        
    return value
