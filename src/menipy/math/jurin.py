import math

def jurin_surface_tension(h_m: float, rho_kg_m3: float, g: float, tube_radius_m: float, contact_angle_rad: float = 0.0) -> float:
    """
    Calculate surface tension using Jurin's Law for capillary rise.
    
    Args:
        h_m: Height of the capillary rise in meters.
        rho_kg_m3: Density of the fluid in kg/m^3.
        g: Acceleration due to gravity in m/s^2.
        tube_radius_m: Radius of the capillary tube in meters.
        contact_angle_rad: Contact angle in radians (default 0.0).
        
    Returns:
        Surface tension (gamma) in N/m.
    """
    if math.isclose(h_m, 0.0):
        return 0.0
    
    gamma = (rho_kg_m3 * g * h_m * tube_radius_m) / (2.0 * math.cos(contact_angle_rad))
    return gamma
