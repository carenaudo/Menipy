import math


def jurin_surface_tension(
    h_m: float,
    rho_kg_m3: float,
    g: float,
    tube_radius_m: float,
    contact_angle_rad: float = 0.0,
) -> float:
    """
    Calculate surface tension using Jurin's Law for capillary rise.

    Academic Reference:
        Jurin, J. (1718). "An account of some experiments shown before the Royal
        Society; with an enquiry into the cause of the ascent and suspension
        of water in capillary tubes." Phil. Trans. R. Soc., 30(355), 739-747.
        DOI: 10.1098/rstl.1717.0026

    Args:
        h_m: Height of the capillary rise in meters.
        rho_kg_m3: Density difference between liquid and gas in kg/m^3.
        g: Acceleration due to gravity in m/s^2.
        tube_radius_m: Radius of the capillary tube in meters.
        contact_angle_rad: Contact angle in radians (default 0.0).

    Returns:
        Surface tension (gamma) in N/m.
    """
    if math.isclose(h_m, 0.0) or tube_radius_m <= 0.0:
        return 0.0

    cos_theta = math.cos(contact_angle_rad)
    if math.isclose(cos_theta, 0.0):
        return 0.0

    gamma = (rho_kg_m3 * g * h_m * tube_radius_m) / (2.0 * cos_theta)
    return max(0.0, gamma)


def jurin_contact_angle(
    h_m: float,
    gamma_n_m: float,
    rho_kg_m3: float,
    g: float,
    tube_radius_m: float,
) -> float:
    """
    Calculate contact angle in radians from capillary rise height and known surface tension.

    Args:
        h_m: Height of the capillary rise in meters.
        gamma_n_m: Surface tension in N/m.
        rho_kg_m3: Density difference in kg/m^3.
        g: Gravity in m/s^2.
        tube_radius_m: Inner radius of the tube in meters.

    Returns:
        Contact angle in radians [0, pi].
    """
    if gamma_n_m <= 0.0 or tube_radius_m <= 0.0:
        return 0.0

    cos_val = (rho_kg_m3 * g * h_m * tube_radius_m) / (2.0 * gamma_n_m)
    cos_val = max(-1.0, min(1.0, cos_val))
    return math.acos(cos_val)


def rayleigh_corrected_capillary_height(
    h_m: float,
    tube_radius_m: float,
) -> float:
    """
    Apply Lord Rayleigh's (1915) meniscus volume correction to capillary rise height.

    For narrow to moderate tubes, the liquid in the meniscus contributes to the
    hydrostatic head. Rayleigh derived the effective height h_eff accounting for
    the meniscus volume:
        h_eff = h + r/3 - 0.1288 * (r^2 / h) + 0.1312 * (r^3 / h^2)

    Academic Reference:
        Rayleigh, Lord (1915). "On the theory of the capillary tube."
        Proc. R. Soc. Lond. A, 92(637), 184-195. DOI: 10.1098/rspa.1915.0006

    Args:
        h_m: Measured height from planar liquid baseline to meniscus apex in meters.
        tube_radius_m: Tube inner radius in meters.

    Returns:
        Effective height h_eff in meters.
    """
    if h_m <= 0.0 or tube_radius_m <= 0.0:
        return max(0.0, h_m)

    r = tube_radius_m
    ratio = r / h_m
    if ratio > 1.0:
        # Meniscus height comparable to rise; fallback to simple 1/3 r correction
        return h_m + r / 3.0

    h_eff = h_m + (r / 3.0) - 0.1288 * (r * ratio) + 0.1312 * (r * ratio**2)
    return max(h_m, h_eff)


def wilhelmy_meniscus_contact_angle(
    h_m: float,
    gamma_n_m: float,
    rho_kg_m3: float,
    g: float,
) -> float:
    """
    Calculate contact angle at a vertical Wilhelmy plate from meniscus height.

    Exact first integral of the 2D Young-Laplace equation:
        h = sqrt(2) * l_c * sqrt(1 - sin(theta))
        where l_c = sqrt(gamma / (rho * g)) is the capillary length.
        sin(theta) = 1 - (rho * g * h^2) / (2 * gamma)

    Args:
        h_m: Meniscus rise height at the plate in meters.
        gamma_n_m: Surface tension in N/m.
        rho_kg_m3: Density difference in kg/m^3.
        g: Gravitational acceleration in m/s^2.

    Returns:
        Contact angle theta in radians [0, pi].
    """
    if gamma_n_m <= 0.0 or rho_kg_m3 <= 0.0:
        return 0.0

    sin_theta = 1.0 - (rho_kg_m3 * g * (h_m**2)) / (2.0 * gamma_n_m)
    sin_theta = max(-1.0, min(1.0, sin_theta))
    return math.asin(sin_theta)

