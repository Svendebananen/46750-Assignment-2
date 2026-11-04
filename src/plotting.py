"""Plots for the hourly results of the three contract modes."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from .data_loader import ContractData
    from .model import Results


def plot_results(result: "Results", contract: "ContractData", output_dir: Path | str) -> list[Path]:
    """Save dispatch and contract-capacity figures for one model result."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    hourly = result.hourly
    hours = np.asarray(hourly.index)
    mode = result.meta.get("contract_mode", "unknown")
    title_mode = {"none": "R: reference case", "individual": "I: individual contracts",
                  "community": "C: community contract"}.get(mode, mode)
    stem = result.question

    dispatch_path = output_dir / f"{stem}_dispatch.png"
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(hours, hourly["load"], label="Load", linewidth=2)
    ax.plot(hours, hourly["pv"], label="PV", linewidth=2)
    ax.plot(hours, hourly["import"], label="Grid import", linewidth=2)
    ax.plot(hours, hourly["export"], label="Grid export", linewidth=2)
    ax.set(
        title=f"{title_mode}: community energy dispatch",
        xlabel="Hour of day",
        ylabel="Energy (kWh)",
        xticks=hours,
    )
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(dispatch_path, dpi=160)
    plt.close(fig)

    contract_path = output_dir / f"{stem}_contract.png"
    net_import = hourly["import"].to_numpy()
    cap = np.asarray(contract.capacity_limit_kW)
    excess = np.maximum(net_import - cap, 0)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(hours, net_import, label="Grid import", linewidth=2)
    ax.plot(hours, cap, "--", label="Contract import cap", linewidth=2)
    if mode == "none":
        ax.text(
            0.02,
            0.96,
            "No contract constraint in R",
            transform=ax.transAxes,
            va="top",
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "0.7"},
        )
    elif np.any(excess > 1e-8):
        ax.fill_between(hours, cap, net_import, where=net_import > cap, alpha=0.25,
                        color="tab:red", label="Import above cap")
    ax.set(
        title=f"{title_mode}: grid import and capacity limit",
        xlabel="Hour of day",
        ylabel="Energy (kWh)",
        xticks=hours,
    )
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(contract_path, dpi=160)
    plt.close(fig)

    return [dispatch_path, contract_path]