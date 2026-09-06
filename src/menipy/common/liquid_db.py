"""Probe liquid database for surface free energy analysis.

Built-in library of standard test liquids with known dispersive and polar
surface tension components, temperature-parameterized per ISO 19403-2 and
DIN 55660-2.  Each entry carries its primary literature citation and DOI.

See ``docs/guides/probe_liquid_reference.md`` for the complete annotated
reference table.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProbeLiquid:
    """A probe liquid with known surface tension components.

    Parameters
    ----------
    name : str
        Human-readable name (e.g. ``"Water"``).
    formula : str
        Chemical formula (Unicode, e.g. ``"H₂O"``).
    cas : str
        CAS Registry Number (e.g. ``"7732-18-5"``).
    gamma_total : float
        Total surface tension γ_L in mN/m.
    gamma_d : float
        Dispersive component γ_L^d in mN/m.
    gamma_p : float
        Polar component γ_L^p in mN/m.
    temperature_c : float
        Measurement temperature in °C.
    reference : str
        Primary literature citation.
    doi : str | None
        Digital Object Identifier for the primary reference.
    notes : str | None
        Handling, safety, or practical measurement notes.
    """

    name: str
    formula: str
    cas: str
    gamma_total: float
    gamma_d: float
    gamma_p: float
    temperature_c: float
    reference: str
    doi: str | None = None
    notes: str | None = None

    @property
    def owrk_x(self) -> float:
        """OWRK plot x-coordinate: √(γ_L^p / γ_L^d)."""
        if self.gamma_d <= 0:
            return 0.0
        return (self.gamma_p / self.gamma_d) ** 0.5

    def validate(self) -> list[str]:
        """Return list of validation warnings (empty if OK)."""
        warnings: list[str] = []
        if self.gamma_total <= 0:
            warnings.append(f"{self.name}: gamma_total must be positive")
        if self.gamma_d < 0:
            warnings.append(f"{self.name}: gamma_d must be non-negative")
        if self.gamma_p < 0:
            warnings.append(f"{self.name}: gamma_p must be non-negative")
        expected = self.gamma_d + self.gamma_p
        if abs(self.gamma_total - expected) > 0.15:
            warnings.append(
                f"{self.name}: gamma_total ({self.gamma_total}) != "
                f"gamma_d + gamma_p ({expected:.1f})"
            )
        return warnings


# ---------------------------------------------------------------------------
# Built-in probe liquid library
# ---------------------------------------------------------------------------
# Canonical entries at 20 °C and 25 °C where available.
# All values in mN/m.  See docs/guides/probe_liquid_reference.md for the
# complete annotated reference with DOIs and handling notes.
# ---------------------------------------------------------------------------

_BUILTIN_LIQUIDS: list[ProbeLiquid] = [
    # ── Water ──────────────────────────────────────────────────────────────
    ProbeLiquid(
        name="Water",
        formula="H₂O",
        cas="7732-18-5",
        gamma_total=72.8,
        gamma_d=21.8,
        gamma_p=51.0,
        temperature_c=20.0,
        reference="Ström et al. (1987)",
        doi="10.1016/0021-9797(87)90043-2",
        notes=(
            "Primary polar standard.  Requires Type 1 ultra-pure water "
            "(18.2 MΩ·cm, TOC < 5 ppb).  Highly sensitive to airborne "
            "surfactant contamination."
        ),
    ),
    ProbeLiquid(
        name="Water",
        formula="H₂O",
        cas="7732-18-5",
        gamma_total=72.0,
        gamma_d=21.8,
        gamma_p=50.2,
        temperature_c=25.0,
        reference="Jasper (1972); Good & van Oss (1991)",
        doi="10.1021/je60054a027",
        notes="25 °C calibration value.",
    ),
    # ── Diiodomethane (methylene iodide) ──────────────────────────────────
    ProbeLiquid(
        name="Diiodomethane",
        formula="CH₂I₂",
        cas="75-11-6",
        gamma_total=50.8,
        gamma_d=50.8,
        gamma_p=0.0,
        temperature_c=20.0,
        reference="Owens & Wendt (1969); Ström et al. (1987)",
        doi="10.1002/app.1969.070130815",
        notes=(
            "Gold standard dispersive liquid (x = 0).  High density "
            "(ρ = 3.325 g/cm³).  Light-sensitive — store in dark amber "
            "bottle over Cu wire."
        ),
    ),
    ProbeLiquid(
        name="Diiodomethane",
        formula="CH₂I₂",
        cas="75-11-6",
        gamma_total=50.0,
        gamma_d=50.0,
        gamma_p=0.0,
        temperature_c=25.0,
        reference="Fowkes (1964); Good & van Oss (1991)",
        doi="10.1021/ie50571a036",
        notes="25 °C calibration value.",
    ),
    # ── Glycerol ──────────────────────────────────────────────────────────
    ProbeLiquid(
        name="Glycerol",
        formula="C₃H₈O₃",
        cas="56-81-5",
        gamma_total=64.0,
        gamma_d=34.0,
        gamma_p=30.0,
        temperature_c=20.0,
        reference="Ström et al. (1987); Busscher et al. (1984)",
        doi="10.1016/0021-9797(87)90043-2",
        notes=(
            "High viscosity (~1400 mPa·s).  Requires slow dosing; highly "
            "hygroscopic — keep tightly sealed."
        ),
    ),
    ProbeLiquid(
        name="Glycerol",
        formula="C₃H₈O₃",
        cas="56-81-5",
        gamma_total=63.4,
        gamma_d=37.0,
        gamma_p=26.4,
        temperature_c=25.0,
        reference="van Oss et al. (1988)",
        doi="10.1016/0009-2614(88)85051-4",
        notes="25 °C alternative calibration.",
    ),
    # ── Ethylene glycol ───────────────────────────────────────────────────
    ProbeLiquid(
        name="Ethylene glycol",
        formula="C₂H₆O₂",
        cas="107-21-1",
        gamma_total=48.0,
        gamma_d=29.0,
        gamma_p=19.0,
        temperature_c=20.0,
        reference="Ström et al. (1987); Kaelble (1970)",
        doi="10.1016/0021-9797(87)90043-2",
        notes=(
            "Widely used secondary polar probe.  Low volatility, moderate "
            "viscosity (~16 mPa·s)."
        ),
    ),
    ProbeLiquid(
        name="Ethylene glycol",
        formula="C₂H₆O₂",
        cas="107-21-1",
        gamma_total=47.7,
        gamma_d=29.3,
        gamma_p=18.4,
        temperature_c=25.0,
        reference="van Oss et al. (1988)",
        doi="10.1016/0009-2614(88)85051-4",
        notes="25 °C alternative calibration.",
    ),
    # ── Formamide ─────────────────────────────────────────────────────────
    ProbeLiquid(
        name="Formamide",
        formula="CH₃NO",
        cas="75-12-7",
        gamma_total=58.0,
        gamma_d=39.5,
        gamma_p=18.5,
        temperature_c=20.0,
        reference="Ström et al. (1987); Owens & Wendt (1969)",
        doi="10.1016/0021-9797(87)90043-2",
        notes=(
            "High dipole moment.  Can dissolve or swell certain polymers "
            "(PMMA, polyamides, polyesters).  Teratogen — handle in fume hood."
        ),
    ),
    # ── DMSO ──────────────────────────────────────────────────────────────
    ProbeLiquid(
        name="DMSO",
        formula="C₂H₆OS",
        cas="67-68-5",
        gamma_total=44.0,
        gamma_d=36.0,
        gamma_p=8.0,
        temperature_c=20.0,
        reference="Janczuk et al. (1993); Wu (1982)",
        doi="10.1016/0021-9797(93)90386-M",
        notes=(
            "Strong organic solvent; readily penetrates skin and swells many "
            "plastics.  Hygroscopic.  Useful for medium-energy polar surfaces."
        ),
    ),
    # ── 1-Bromonaphthalene ────────────────────────────────────────────────
    ProbeLiquid(
        name="1-Bromonaphthalene",
        formula="C₁₀H₇Br",
        cas="90-11-9",
        gamma_total=44.4,
        gamma_d=44.4,
        gamma_p=0.0,
        temperature_c=20.0,
        reference="Fowkes (1964); Ström et al. (1987)",
        doi="10.1021/ie50571a036",
        notes=(
            "Purely dispersive alternative to diiodomethane (x = 0).  Useful "
            "if diiodomethane reacts with the sample."
        ),
    ),
    ProbeLiquid(
        name="1-Bromonaphthalene",
        formula="C₁₀H₇Br",
        cas="90-11-9",
        gamma_total=44.6,
        gamma_d=44.6,
        gamma_p=0.0,
        temperature_c=25.0,
        reference="Good & van Oss (1991)",
        doi="10.1007/978-1-4615-3106-2_7",
        notes="25 °C calibration value.",
    ),
    # ── Hexadecane ────────────────────────────────────────────────────────
    ProbeLiquid(
        name="Hexadecane",
        formula="C₁₆H₃₄",
        cas="544-76-3",
        gamma_total=27.5,
        gamma_d=27.5,
        gamma_p=0.0,
        temperature_c=20.0,
        reference="Jasper (1972); Fowkes (1964)",
        doi="10.1021/je60054a027",
        notes=(
            "Purely dispersive alkane (x = 0).  Low surface tension; will "
            "completely wet surfaces with γ_S > 27.5 mN/m."
        ),
    ),
    # ── Thiodiglycol ──────────────────────────────────────────────────────
    ProbeLiquid(
        name="Thiodiglycol",
        formula="C₄H₁₀O₂S",
        cas="111-48-8",
        gamma_total=54.0,
        gamma_d=14.8,
        gamma_p=39.2,
        temperature_c=20.0,
        reference="Berger (1991); Janczuk et al. (1993)",
        doi="10.1016/0021-9797(93)90386-M",
        notes="High viscosity (~65 mPa·s).  Fowkes/OWRK partitioning convention.",
    ),
    # ── Tricresyl phosphate ───────────────────────────────────────────────
    ProbeLiquid(
        name="Tricresyl phosphate",
        formula="C₂₁H₂₁O₄P",
        cas="1330-78-5",
        gamma_total=40.9,
        gamma_d=39.2,
        gamma_p=1.7,
        temperature_c=20.0,
        reference="Panzer (1973); Wu (1982)",
        doi="10.1016/0021-9797(73)90004-0",
        notes=(
            "Predominantly dispersive aromatic ester with low polar component.  "
            "High boiling point, low volatility, neurotoxic plasticizer."
        ),
    ),
    # ── Benzyl alcohol ────────────────────────────────────────────────────
    ProbeLiquid(
        name="Benzyl alcohol",
        formula="C₇H₈O",
        cas="100-51-6",
        gamma_total=39.0,
        gamma_d=29.0,
        gamma_p=10.0,
        temperature_c=20.0,
        reference="Kaelble (1970); Panzer (1973)",
        doi="10.1016/S0021-9797(70)80035-9",
        notes=(
            "Moderately polar aromatic alcohol.  Low volatility, moderate "
            "viscosity.  Alternative probe for mid-polarity surfaces."
        ),
    ),
]


def get_builtin_liquids() -> list[ProbeLiquid]:
    """Return the complete built-in probe liquid library."""
    return list(_BUILTIN_LIQUIDS)


def get_liquid(
    name: str,
    temperature_c: float = 20.0,
    *,
    tolerance_c: float = 1.0,
) -> ProbeLiquid | None:
    """Look up a probe liquid by name and temperature.

    Parameters
    ----------
    name : str
        Liquid name (case-insensitive, underscore/hyphen normalised).
    temperature_c : float
        Target temperature in °C (default 20.0).
    tolerance_c : float
        Acceptable temperature deviation in °C (default 1.0).

    Returns
    -------
    ProbeLiquid | None
        Best matching entry, or ``None`` if not found.
    """
    normalised = name.strip().lower().replace("_", " ").replace("-", " ")
    candidates: list[ProbeLiquid] = []
    for liq in _BUILTIN_LIQUIDS:
        liq_norm = liq.name.lower().replace("_", " ").replace("-", " ")
        if liq_norm == normalised and abs(liq.temperature_c - temperature_c) <= tolerance_c:
            candidates.append(liq)
    if not candidates:
        # Relaxed: try partial match
        for liq in _BUILTIN_LIQUIDS:
            liq_norm = liq.name.lower().replace("_", " ").replace("-", " ")
            if normalised in liq_norm and abs(liq.temperature_c - temperature_c) <= tolerance_c:
                candidates.append(liq)
    if not candidates:
        return None
    # Pick closest temperature
    candidates.sort(key=lambda liq: abs(liq.temperature_c - temperature_c))
    return candidates[0]


def list_liquid_names() -> list[str]:
    """Return sorted unique liquid names from the built-in library."""
    return sorted({liq.name for liq in _BUILTIN_LIQUIDS})


def format_liquid_table(
    temperature_c: float | None = None,
) -> str:
    """Format the built-in library as a human-readable CLI table.

    Parameters
    ----------
    temperature_c : float | None
        If given, filter to entries at this temperature (±1 °C).
        If ``None``, show all entries.
    """
    entries = _BUILTIN_LIQUIDS
    if temperature_c is not None:
        entries = [liq for liq in entries if abs(liq.temperature_c - temperature_c) <= 1.0]

    header = (
        f"{'Liquid':<25} {'T(C)':>6} {'g_total':>8} {'g_d':>8} {'g_p':>8} "
        f"{'x_OWRK':>7}  {'Reference'}"
    )
    sep = "-" * len(header)
    lines = [header, sep]
    for liq in entries:
        lines.append(
            f"{liq.name:<25} {liq.temperature_c:>6.1f} {liq.gamma_total:>8.1f} "
            f"{liq.gamma_d:>8.1f} {liq.gamma_p:>8.1f} {liq.owrk_x:>7.3f}  {liq.reference}"
        )
    return "\n".join(lines)
