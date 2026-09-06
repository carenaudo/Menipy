"""Interfacial dilational rheology and droplet oscillation mathematical formulations.

Academic References:
    1. Lucassen-Reynders, E. H., & Lucassen, J. (1969).
       "Properties of capillary waves."
       Advances in Colloid and Interface Science, 2(4), 347-395.
       DOI: 10.1016/0001-8686(69)80006-0

    2. Loglio, G., Tesei, U., & Cini, R. (1988).
       "Measurement of interfacial dilatational properties by a dynamic method."
       Journal of Colloid and Interface Science, 126(2), 486-492.
       DOI: 10.1016/0021-9797(88)90150-6

    3. Miller, R., et al. (2000).
       "Interfacial dilatational rheology by oscillating bubble/drop methods."
       Colloids and Surfaces A, 175(1-2), 125-134.
       DOI: 10.1016/S0927-7757(00)00525-7

    4. Rayleigh, Lord (1879).
       "On the capillary phenomena of jets."
       Proceedings of the Royal Society of London, 29(196-199), 71-97.
       DOI: 10.1098/rspl.1879.0015

Attribution & Clean-Room Implementation:
    Independent clean-room Python/NumPy implementation authored for Menipy under MIT license.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import curve_fit


def harmonic_sine_fit(
    time_s: np.ndarray,
    signal: np.ndarray,
    frequency_hz: float | None = None,
) -> tuple[float, float, float, float]:
    """
    Fit a harmonic sinusoidal model: y(t) = y0 + A * sin(2*pi*f*t + phi).

    Args:
        time_s: 1D array of timestamps in seconds.
        signal: 1D array of signal values (e.g. area, surface tension, radius).
        frequency_hz: Known frequency in Hz (if None, estimated via FFT).

    Returns:
        tuple (y0, amplitude, frequency_hz, phase_rad)
    """
    t = np.asarray(time_s, dtype=float)
    y = np.asarray(signal, dtype=float)

    if len(t) < 4 or float(np.ptp(t)) <= 0:
        return float(np.mean(y)), 0.0, float(frequency_hz or 1.0), 0.0

    y0_init = float(np.mean(y))
    amp_init = float((np.max(y) - np.min(y)) / 2.0)

    # Estimate frequency if not provided
    if frequency_hz is None or frequency_hz <= 0:
        dt = float(np.mean(np.diff(t)))
        if dt > 0 and len(y) >= 8:
            n = int(2 ** np.ceil(np.log2(len(y))))
            fft_mag = np.abs(np.fft.rfft(y - y0_init, n=n))
            freqs = np.fft.rfftfreq(n, d=dt)
            if len(freqs) > 1:
                fft_mag[0] = 0.0
                k = int(np.argmax(fft_mag))
                f_init = float(freqs[k])
            else:
                f_init = 1.0
        else:
            f_init = 1.0
    else:
        f_init = float(frequency_hz)

    # Nonlinear fit
    try:
        def model(t_arr, y0, amp, f, phi):
            return y0 + amp * np.sin(2.0 * np.pi * f * t_arr + phi)

        p0 = [y0_init, amp_init, f_init, 0.0]
        bounds = (
            [-np.inf, 0.0, max(1e-3, f_init * 0.5), -np.pi],
            [np.inf, np.inf, f_init * 1.5 + 10.0, np.pi],
        )
        popt, _ = curve_fit(model, t, y, p0=p0, bounds=bounds, maxfev=1000)
        y0_fit, amp_fit, f_fit, phi_fit = popt
        return float(y0_fit), float(amp_fit), float(f_fit), float(phi_fit)
    except Exception:
        return y0_init, amp_init, f_init, 0.0


def dilational_viscoelasticity(
    A0: float,
    delta_A: float,
    delta_gamma: float,
    phase_shift_rad: float,
    frequency_hz: float,
) -> dict[str, float]:
    """
    Compute interfacial dilational rheology viscoelastic moduli.

    Formulation:
        |E| = A0 * (delta_gamma / delta_A)
        E'  = |E| * cos(delta)   (Storage modulus / Elasticity)
        E'' = |E| * sin(delta)   (Loss modulus / Viscous dissipation)
        eta_d = E'' / (2 * pi * f) (Dilational surface viscosity)

    Args:
        A0: Mean interfacial area in mm^2 or m^2.
        delta_A: Area perturbation amplitude in same units as A0.
        delta_gamma: Surface tension oscillation amplitude in mN/m.
        phase_shift_rad: Phase shift delta = phi_gamma - phi_A in radians.
        frequency_hz: Oscillation frequency in Hz.

    Returns:
        Dictionary containing:
            modulus_E: Complex modulus magnitude |E| (mN/m).
            storage_modulus_E_prime: Dilational elasticity E' (mN/m).
            loss_modulus_E_double_prime: Dilational loss modulus E'' (mN/m).
            dilational_viscosity: Surface dilational viscosity eta_d (mN*s/m).
            phase_shift_deg: Phase shift in degrees.
    """
    if delta_A <= 1e-12 or A0 <= 0:
        return {
            "modulus_E": 0.0,
            "storage_modulus_E_prime": 0.0,
            "loss_modulus_E_double_prime": 0.0,
            "dilational_viscosity": 0.0,
            "phase_shift_deg": math.degrees(phase_shift_rad),
        }

    relative_area_strain = delta_A / A0
    modulus_E = delta_gamma / relative_area_strain

    e_prime = modulus_E * math.cos(phase_shift_rad)
    e_double_prime = modulus_E * math.sin(phase_shift_rad)

    omega = 2.0 * math.pi * max(1e-4, frequency_hz)
    dilational_viscosity = e_double_prime / omega

    return {
        "modulus_E": float(modulus_E),
        "storage_modulus_E_prime": float(e_prime),
        "loss_modulus_E_double_prime": float(e_double_prime),
        "dilational_viscosity": float(dilational_viscosity),
        "phase_shift_deg": float(math.degrees(phase_shift_rad)),
    }


def rayleigh_lamb_surface_tension(
    frequency_hz: float,
    radius_m: float,
    rho_drop_kg_m3: float,
    rho_medium_kg_m3: float = 1.2,
    mode_n: int = 2,
) -> float:
    """
    Calculate surface tension from droplet natural oscillation frequency (Rayleigh-Lamb).

    Formula:
        omega_n^2 = [ n * (n - 1) * (n + 2) * gamma ] / [ rho_1 * R^3 + (n / (n + 1)) * rho_2 * R^3 ]
        For fundamental mode n=2 (quadrupole):
        omega_2^2 = (24 * gamma) / [ 3 * rho_1 * R^3 + 2 * rho_2 * R^3 ]
        gamma = [ (3 * rho_1 + 2 * rho_2) / 24 ] * (2 * pi * f_2)^2 * R^3

    Args:
        frequency_hz: Measured oscillation frequency f_n in Hz.
        radius_m: Equilibrium equivalent spherical radius R0 in meters.
        rho_drop_kg_m3: Liquid droplet density in kg/m^3.
        rho_medium_kg_m3: Ambient fluid density in kg/m^3 (default 1.2 for air).
        mode_n: Oscillation mode number (default 2 for fundamental quadrupole).

    Returns:
        Surface tension gamma in N/m.
    """
    if frequency_hz <= 0 or radius_m <= 0 or rho_drop_kg_m3 <= 0:
        return 0.0

    omega = 2.0 * math.pi * frequency_hz
    n = max(2, int(mode_n))

    numerator_factor = n * (n - 1) * (n + 2)
    effective_inertia = rho_drop_kg_m3 + (float(n) / float(n + 1)) * rho_medium_kg_m3

    gamma = (omega**2 * effective_inertia * (radius_m**3)) / float(numerator_factor)
    return float(gamma)
