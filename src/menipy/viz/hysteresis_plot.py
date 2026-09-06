"""Visualization of contact angle hysteresis from needle-in-sessile-drop sequences.

Generates:
1. **Dynamic Timeline Plot**: Contact angle and base diameter vs. time/frame
   with state-colored plateau bands.
2. **Hysteresis Loop Plot**: Contact angle vs. base diameter (theta vs. d)
   showing advancing and receding cycles.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from menipy.common.needle_hysteresis import NeedleHysteresisResult

logger = logging.getLogger(__name__)


def plot_hysteresis_timeline(
    result: NeedleHysteresisResult,
    output_path: Path | None = None,
    *,
    title: str = "Contact Angle Hysteresis — Timeline",
    dpi: int = 150,
) -> Figure:
    """Generate a dual-axis timeline chart: contact angles and base diameter vs time."""
    import matplotlib.pyplot as plt

    frames = [f for f in result.frames if f.accepted]
    if not frames:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No valid frames available for timeline plot", ha="center", va="center")
        if output_path:
            fig.savefig(str(output_path), dpi=dpi)
        return fig

    t = [f.timestamp_s for f in frames]
    theta_mean = [f.theta_mean_deg for f in frames]
    theta_l = [f.theta_left_deg for f in frames]
    theta_r = [f.theta_right_deg for f in frames]
    diam = [f.base_diameter_mm for f in frames]
    states = [f.state for f in frames]

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax2 = ax1.twinx()

    # 1. State background shading
    # Group contiguous state intervals
    idx = 0
    while idx < len(frames):
        st = states[idx]
        end = idx + 1
        while end < len(frames) and states[end] == st:
            end += 1
        t_start = t[idx]
        t_end = t[end - 1]
        if st == "advancing":
            ax1.axvspan(t_start, t_end, color="#10B981", alpha=0.18, label="Advancing" if idx == 0 else None)
        elif st == "receding":
            ax1.axvspan(t_start, t_end, color="#3B82F6", alpha=0.18, label="Receding" if idx == 0 else None)
        idx = end

    # 2. Contact angle curves on ax1
    ax1.plot(t, theta_mean, color="#1E293B", linewidth=2.0, label=r"$\theta_{\mathrm{mean}}$")
    ax1.scatter(t, theta_l, color="#DC2626", s=18, alpha=0.6, label=r"$\theta_{\mathrm{left}}$")
    ax1.scatter(t, theta_r, color="#7C3AED", s=18, alpha=0.6, label=r"$\theta_{\mathrm{right}}$")

    # Advancing / receding plateau lines
    if "theta_advancing_deg" in result.summary:
        th_a = result.summary["theta_advancing_deg"]
        ax1.axhline(th_a, color="#059669", linestyle="--", linewidth=1.5, label=f"$\\theta_A = {th_a:.1f}^\\circ$")
    if "theta_receding_deg" in result.summary:
        th_r = result.summary["theta_receding_deg"]
        ax1.axhline(th_r, color="#2563EB", linestyle="--", linewidth=1.5, label=f"$\\theta_R = {th_r:.1f}^\\circ$")

    ax1.set_xlabel("Time (s)", fontsize=11)
    ax1.set_ylabel(r"Contact Angle $\theta$ (deg)", fontsize=11, color="#1E293B")
    ax1.tick_params(axis="y", labelcolor="#1E293B")
    ax1.set_ylim(bottom=0)

    # 3. Base diameter on ax2
    valid_diam = [d for d in diam if d is not None]
    if valid_diam:
        ax2.plot(t, diam, color="#D97706", linewidth=1.8, linestyle="-.", label="Base diameter")
        ax2.set_ylabel("Base Diameter (mm)", fontsize=11, color="#D97706")
        ax2.tick_params(axis="y", labelcolor="#D97706")

    # Summary box
    s = result.summary
    text_lines = []
    if "theta_advancing_deg" in s:
        text_lines.append(f"$\\theta_A$ = {s['theta_advancing_deg']:.1f}°")
    if "theta_receding_deg" in s:
        text_lines.append(f"$\\theta_R$ = {s['theta_receding_deg']:.1f}°")
    if "contact_angle_hysteresis_deg" in s:
        text_lines.append(f"$\\Delta\\theta$ = {s['contact_angle_hysteresis_deg']:.1f}°")

    if text_lines:
        ax1.text(
            0.03,
            0.05,
            "\n".join(text_lines),
            transform=ax1.transAxes,
            fontsize=10,
            verticalalignment="bottom",
            bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "edgecolor": "#CBD5E1", "alpha": 0.9},
        )

    ax1.set_title(title, fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.3, linestyle=":")

    # Merge legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    # Deduplicate labels
    by_label = dict(zip(labels1 + labels2, lines1 + lines2))
    ax1.legend(by_label.values(), by_label.keys(), loc="upper right", fontsize=8)

    fig.tight_layout()
    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(output_path), dpi=dpi)
        logger.info(f"Timeline plot saved to {output_path}")

    return fig


def plot_hysteresis_loop(
    result: NeedleHysteresisResult,
    output_path: Path | None = None,
    *,
    title: str = "Contact Angle vs. Base Diameter (Hysteresis Loop)",
    dpi: int = 150,
) -> Figure:
    """Generate the signature contact angle vs. base diameter hysteresis loop."""
    import matplotlib.pyplot as plt

    frames = [f for f in result.frames if f.accepted and f.base_diameter_mm is not None]
    if not frames:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.text(0.5, 0.5, "No valid data for hysteresis loop", ha="center", va="center")
        if output_path:
            fig.savefig(str(output_path), dpi=dpi)
        return fig

    d = np.array([f.base_diameter_mm for f in frames])
    th = np.array([f.theta_mean_deg for f in frames])
    states = [f.state for f in frames]

    fig, ax = plt.subplots(figsize=(7, 5))

    # Connect points chronologically with thin gray line
    ax.plot(d, th, color="#94A3B8", linewidth=1.0, zorder=1)

    # Color scatter points by state
    color_map = {"advancing": "#10B981", "receding": "#3B82F6", "pinned": "#64748B"}
    colors = [color_map.get(st, "#94A3B8") for st in states]
    ax.scatter(d, th, c=colors, s=35, zorder=2, edgecolors="white", linewidths=0.5)

    # Annotate direction
    if len(d) > 8:
        mid = len(d) // 4
        ax.annotate(
            "",
            xy=(d[mid + 2], th[mid + 2]),
            xytext=(d[mid], th[mid]),
            arrowprops={"arrowstyle": "->", "color": "#0F172A", "lw": 1.5},
        )

    # Plateau lines
    if "theta_advancing_deg" in result.summary:
        th_a = result.summary["theta_advancing_deg"]
        ax.axhline(th_a, color="#059669", linestyle="--", label=f"$\\theta_A$ = {th_a:.1f}°")
    if "theta_receding_deg" in result.summary:
        th_r = result.summary["theta_receding_deg"]
        ax.axhline(th_r, color="#2563EB", linestyle="--", label=f"$\\theta_R$ = {th_r:.1f}°")

    ax.set_xlabel("Base Contact Diameter (mm)", fontsize=11)
    ax.set_ylabel(r"Contact Angle $\theta$ (deg)", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle=":")
    ax.legend(loc="best", fontsize=9)

    fig.tight_layout()
    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(output_path), dpi=dpi)
        logger.info(f"Hysteresis loop plot saved to {output_path}")

    return fig
