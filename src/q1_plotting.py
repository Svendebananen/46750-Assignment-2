"""Figures for the Question 1 contract sensitivity and settlement experiments."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


def _zero_crossing(x, y):
    """Return the first linearly interpolated zero crossing, if one exists."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    for k in range(len(x) - 1):
        if y[k] * y[k + 1] < 0:
            return float(x[k] - y[k] * (x[k + 1] - x[k]) / (y[k + 1] - y[k]))
    return None


def plot_gain_map(grid, output_dir: Path | str, base: tuple[float, float] = (0.75, 1.3)) -> Path:
    """Plot community gains of I and C relative to R over discount and cap factor."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    limit = float(max(np.abs(grid["gain_I_vs_R"]).max(), np.abs(grid["gain_C_vs_R"]).max()))
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, column, title in zip(axes, ("gain_I_vs_R", "gain_C_vs_R"), ("I vs R", "C vs R")):
        table = grid.pivot(index="beta", columns="phi", values=column).sort_index(ascending=False)
        image = ax.imshow(table.values, cmap="RdYlGn", vmin=-limit, vmax=limit, aspect="auto")
        ax.set_xticks(range(len(table.columns)), [f"{value:g}" for value in table.columns])
        ax.set_yticks(range(len(table.index)), [f"{value:g}" for value in table.index])
        ax.set(
            title=f"{title}: community gain (DKK/day)",
            xlabel="cap factor $\\phi$",
            ylabel="$\\beta^{DSO}$",
        )
        for (row, col), value in np.ndenumerate(table.values):
            ax.text(col, row, f"{value:.1f}", ha="center", va="center", fontsize=8)
        if table.values.min() < 0 < table.values.max():
            ax.contour(table.values, levels=[0.0], colors="black", linewidths=1.6)
        betas, phis = list(table.index), list(table.columns)
        if base[0] in betas and base[1] in phis:
            ax.add_patch(
                Rectangle(
                    (phis.index(base[1]) - 0.5, betas.index(base[0]) - 0.5),
                    1,
                    1,
                    fill=False,
                    edgecolor="black",
                    linewidth=2.8,
                )
            )
        fig.colorbar(image, ax=ax)
    fig.tight_layout()
    path = output_dir / "q1_gain_map.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_dso_view(grid, output_dir: Path | str, beta: float = 0.75, base_phi: float = 1.3) -> Path:
    """Plot DSO revenue forgone per kWh of exceedance removed for I and C."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = grid[np.isclose(grid["beta"], beta)].sort_values("phi")
    phi = selected["phi"].to_numpy()
    colors = {"I": "tab:orange", "C": "tab:blue"}
    fig, ax = plt.subplots(figsize=(6.2, 3.9))
    for key in ("I", "C"):
        removed = selected["exc_R"].to_numpy() - selected[f"exc_{key}"].to_numpy()
        revenue_change = selected[f"rev_{key}"].to_numpy() - selected["rev_R"].to_numpy()
        valid = removed > 1.0
        per_kwh = np.where(valid, -revenue_change / np.where(valid, removed, 1.0), np.nan)
        ax.plot(phi, per_kwh, marker="o", color=colors[key], label=key)
        crossing = _zero_crossing(phi, selected[f"gain_{key}_vs_R"].to_numpy())
        if crossing is not None:
            ax.axvline(crossing, color=colors[key], linestyle=":", linewidth=1.4)
            ax.annotate(
                f"{key} breaks even\n$\\phi\\approx{crossing:.2f}$",
                (crossing, 0),
                xytext=(4, 12 if key == "C" else 36),
                textcoords="offset points",
                fontsize=8,
                color=colors[key],
            )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(base_phi, color="gray", linestyle="--", linewidth=1.0)
    ax.set(
        xlabel="cap factor $\\phi$",
        ylabel="DKK per kWh removed",
        title="Revenue the DSO gives up per kWh of exceedance removed",
    )
    ax.grid(alpha=0.25)
    ax.legend(title=f"vs R, $\\beta^{{DSO}}={beta:g}$", fontsize=8)
    fig.tight_layout()
    path = output_dir / "q1_dso_view.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def plot_member_gains(members, output_dir: Path | str) -> Path:
    """Plot member gains under the C settlement relative to R and I."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    positions = np.arange(len(members))
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.bar(positions - 0.2, members["gain_vs_R"], 0.4, label="vs. R")
    ax.bar(positions + 0.2, members["gain_vs_I"], 0.4, label="vs. I")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(positions, members.index, rotation=70)
    ax.set(ylabel="Gain (DKK/day)", title="Settlement at $\\lambda_t$ in C (manager keeps the surplus)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path = output_dir / "q1_member_gains.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


class Question1Plotter:
    """Create Question 1 figures in one output directory."""

    def __init__(self, output_dir: Path | str):
        self.output_dir = Path(output_dir)

    def gain_map(self, grid) -> Path:
        return plot_gain_map(grid, self.output_dir)

    def dso_view(self, grid) -> Path:
        return plot_dso_view(grid, self.output_dir)

    def member_gains(self, members) -> Path:
        return plot_member_gains(members, self.output_dir)
