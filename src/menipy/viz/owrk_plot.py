"""OWRK regression plot for surface free energy analysis.

Generates a scatter plot of linearised OWRK data points with the OLS
regression line and 95 % confidence band.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from menipy.math.surface_energy import OWRKResult

logger = logging.getLogger(__name__)


def plot_owrk(
    result: OWRKResult,
    output_path: Path | None = None,
    *,
    title: str = "OWRK Surface Energy Analysis",
    show: bool = False,
    dpi: int = 150,
) -> Figure:
    """Create the OWRK regression plot.

    Parameters
    ----------
    result : OWRKResult
        OWRK computation result with data points and regression parameters.
    output_path : Path | None
        If given, save the figure as PNG to this path.
    title : str
        Plot title.
    show : bool
        If ``True``, call ``plt.show()`` (blocks in non-interactive mode).
    dpi : int
        Resolution for saved PNG.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))

    x = np.array(result.x_coords)
    y = np.array(result.y_coords)

    # Scatter points with labels
    ax.scatter(x, y, color="#2563EB", s=80, zorder=5, edgecolors="white", linewidths=0.8)

    for xi, yi, name in zip(x, y, result.liquid_names):
        ax.annotate(
            name,
            (xi, yi),
            textcoords="offset points",
            xytext=(8, 6),
            fontsize=9,
            color="#374151",
        )

    # Regression line
    x_line = np.linspace(0, max(x.max() * 1.15, 1.8), 100)
    y_line = result.slope * x_line + result.intercept
    ax.plot(x_line, y_line, color="#DC2626", linewidth=1.8, label="OWRK fit", zorder=4)

    # 95% confidence band (only for N >= 3)
    if result.r_squared is not None and result.se_slope is not None and len(x) >= 3:
        from scipy.stats import t as t_dist

        n = len(x)
        x_mean = np.mean(x)
        ss_xx = np.sum((x - x_mean) ** 2)
        s_yx = np.sqrt(np.sum(np.array(result.residuals) ** 2) / (n - 2))

        t_crit = t_dist.ppf(0.975, df=n - 2)

        se_line = s_yx * np.sqrt(1.0 / n + (x_line - x_mean) ** 2 / ss_xx)
        y_upper = result.slope * x_line + result.intercept + t_crit * se_line
        y_lower = result.slope * x_line + result.intercept - t_crit * se_line

        ax.fill_between(
            x_line,
            y_lower,
            y_upper,
            alpha=0.12,
            color="#DC2626",
            label="95% CI",
            zorder=3,
        )

    # Annotation box with results
    text_lines = [
        f"γ$_S^d$ = {result.gamma_s_d:.1f} mN/m",
        f"γ$_S^p$ = {result.gamma_s_p:.1f} mN/m",
        f"γ$_S$   = {result.gamma_s_total:.1f} mN/m",
    ]
    if result.r_squared is not None:
        text_lines.append(f"R² = {result.r_squared:.4f}")
    annotation_text = "\n".join(text_lines)

    ax.text(
        0.97,
        0.03,
        annotation_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox={"boxstyle": "round,pad=0.5", "facecolor": "white", "edgecolor": "#D1D5DB", "alpha": 0.9},
        fontfamily="monospace",
    )

    # Axis labels
    ax.set_xlabel(
        r"$\sqrt{\gamma_L^p\, /\, \gamma_L^d}$",
        fontsize=12,
    )
    ax.set_ylabel(
        r"$\gamma_L\,(1 + \cos\theta)\, /\, (2\sqrt{\gamma_L^d})$"
        r"  $[\sqrt{\mathrm{mN/m}}]$",
        fontsize=12,
    )
    ax.set_title(title, fontsize=13, fontweight="bold")

    ax.set_xlim(left=-0.05)
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3, linestyle="--")

    fig.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(output_path), dpi=dpi, bbox_inches="tight")
        logger.info(f"OWRK plot saved to {output_path}")

    if show:
        plt.show()

    return fig
