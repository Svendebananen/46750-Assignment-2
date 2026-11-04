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


def plot_results(
    result: "Results",
    contract: "ContractData",
    energy_price: np.ndarray,
    output_dir: Path | str,
) -> list[Path]:
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
        title=f"{title_mode}: Accumulated community energy dispatch",
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
    handles, labels = ax.get_legend_handles_labels()
    price_ax = ax.twinx()
    price_line, = price_ax.plot(
        hours,
        energy_price,
        color="tab:purple",
        linewidth=2,
        label="Day-ahead price",
    )
    price_ax.set_ylabel("Day-ahead price (DKK/kWh)", color="tab:purple")
    price_ax.tick_params(axis="y", labelcolor="tab:purple")
    handles.append(price_line)
    labels.append("Day-ahead price")
    ax.legend(handles, labels)
    fig.tight_layout()
    fig.savefig(contract_path, dpi=160)
    plt.close(fig)

    return [dispatch_path, contract_path]


def plot_member_operating_Costs(
    reference: "Results",
    contract_result: "Results",
    members: list,
    output_dir: Path | str,
) -> Path:
    """Plot each contract's operating-cost compared to the reference, stacked by member and asset type."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    asset_types = ("No asset", "PV", "Battery", "PV and battery")
    asset_members = {
        asset_type: [
            member for member in members
            if (
                asset_type == "No asset" and not member.has_pv and not member.has_battery
                or asset_type == "PV" and member.has_pv and not member.has_battery
                or asset_type == "Battery" and not member.has_pv and member.has_battery
                or asset_type == "PV and battery" and member.has_pv and member.has_battery
            )
        ]
        for asset_type in asset_types
    }
    reference_costs = reference.meta["member_operating_costs"]
    contract_costs = contract_result.meta["member_operating_costs"]
    mode = contract_result.meta["contract_mode"]
    asset_member_ids = {
        asset_type: {member.consumer_id for member in asset_members[asset_type]}
        for asset_type in asset_types
    }
    positions = np.arange(len(asset_types))
    width = 0.65
    fig, ax = plt.subplots(figsize=(10, 6))
    bottoms = np.zeros(len(asset_types))
    for member in members:
        operating_Costs = np.array([
            reference_costs[member.consumer_id] - contract_costs[member.consumer_id]
            if member.consumer_id in asset_member_ids[asset_type] else 0.0
            for asset_type in asset_types
        ])
        if not np.any(operating_Costs):
            continue
        ax.bar(
            positions,
            operating_Costs,
            width,
            bottom=bottoms,
            label=member.consumer_id,
        )
        bottoms += operating_Costs
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set(
        title=f"{mode.upper()} contract: operating-cost comparison to reference R",
        ylabel="Operating cost versus reference R (DKK/member)",
        xticks=positions,
        xticklabels=asset_types,
    )
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="Member", ncol=2)
    fig.tight_layout()
    path = output_dir / f"{contract_result.question}_member_operating_Costs.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_member_profits(
    result: "Results",
    members: list,
    output_dir: Path | str,
) -> Path:
    """Plot each member's full-day profit, stacked by member and asset type."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    asset_types = ("No asset", "PV", "Battery", "PV and battery")
    asset_member_ids = {
        asset_type: {
            member.consumer_id for member in members
            if (
                asset_type == "No asset" and not member.has_pv and not member.has_battery
                or asset_type == "PV" and member.has_pv and not member.has_battery
                or asset_type == "Battery" and not member.has_pv and member.has_battery
                or asset_type == "PV and battery" and member.has_pv and member.has_battery
            )
        }
        for asset_type in asset_types
    }
    profits = result.meta["member_profits"]
    positions = np.arange(len(asset_types))
    bottoms = np.zeros(len(asset_types))
    fig, ax = plt.subplots(figsize=(10, 6))
    for member in members:
        values = np.array([
            profits[member.consumer_id]
            if member.consumer_id in asset_member_ids[asset_type] else 0.0
            for asset_type in asset_types
        ])
        if not np.any(values):
            continue
        ax.bar(positions, values, 0.65, bottom=bottoms, label=member.consumer_id)
        bottoms += values
    ax.axhline(0, color="black", linewidth=0.8)
    mode = result.meta["contract_mode"]
    ax.set(
        title=f"{mode.upper()} contract: member profit by asset type",
        ylabel="Profit (DKK/day)",
        xticks=positions,
        xticklabels=asset_types,
    )
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="Member", ncol=2)
    fig.tight_layout()
    path = output_dir / f"{result.question}_member_profits.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path