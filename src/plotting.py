"""Plots for the hourly results of the three contract modes."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .q2_plotting import (
    plot_dp_benefits_by_member,
    plot_dp_cap_sweep,
    plot_dp_compare_prices,
    plot_dp_cost_of_requirements,
    plot_dp_prices,
    plot_dp_shift_heatmap,
)

__all__ = [
    "plot_results",
    "plot_member_benefits",
    "plot_dp_benefits_by_member",
    "plot_dp_cap_sweep",
    "plot_dp_compare_prices",
    "plot_dp_cost_of_requirements",
    "plot_dp_prices",
    "plot_dp_shift_heatmap",
]

if TYPE_CHECKING:
    from .data_loader import ContractData
    from .model import Results


def plot_results(
    result: "Results",
    contract: "ContractData",
    energy_price: np.ndarray,
    output_dir: Path | str,
    tag: str = "",
) -> list[Path]:
    """Save dispatch and contract-capacity figures for one model result."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    hourly = result.hourly
    hours = np.asarray(hourly.index)
    mode = result.meta.get("contract_mode", "unknown")
    title_mode = {
        "none": "R: reference case",
        "individual": "I: individual contracts",
        "community": "C: community contract",
        "dynamic": "DP: dynamic pricing",
    }.get(mode, mode)
    stem = result.question + (f"_{tag}" if tag else "")

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
        ax.fill_between(
            hours,
            cap,
            net_import,
            where=net_import > cap,
            alpha=0.25,
            color="tab:red",
            label="Import above cap",
        )
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


def plot_member_benefits(
    reference: "Results",
    contract_result: "Results",
    members: list,
    output_dir: Path | str,
) -> Path:
    """Plot each contract's operating-cost benefit, stacked by member and asset type."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    asset_types = ("No asset", "PV", "Battery", "PV and battery")
    asset_members = {
        asset_type: [
            member
            for member in members
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
        benefits = np.array(
            [
                reference_costs[member.consumer_id] - contract_costs[member.consumer_id]
                if member.consumer_id in asset_member_ids[asset_type]
                else 0.0
                for asset_type in asset_types
            ]
        )
        if not np.any(benefits):
            continue
        ax.bar(
            positions,
            benefits,
            width,
            bottom=bottoms,
            label=member.consumer_id,
        )
        bottoms += benefits
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set(
        title=f"{mode.upper()} contract: operating-cost benefit by asset type",
        ylabel="Benefit versus reference R (DKK/member)",
        xticks=positions,
        xticklabels=asset_types,
    )
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="Member", ncol=2)
    fig.tight_layout()
    path = output_dir / f"{contract_result.question}_member_benefits.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_q3_validation(scenarios, base_members, cap, in_model_cost, evaluator_cost, dp_cost: float,
                       output_dir: Path | str, tag: str = "q3_validation") -> Path:
    """Question 3, validation in one figure.
    (a) What the uncertainty does: the community's net reference demand (sum of the members' reference loads minus
        their available PV) in every in-sample scenario, as a band (min-max over the scenarios) around the base case,
        against the import cap. Net of PV so that it is comparable with the cap.
    (b) That the model holds: each scenario's community cost in the SDP model (bars) and when its prices are replayed
        by the out-of-sample evaluator (markers), against the deterministic DP on the expected parameters (line)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def net_demand(members):
        return sum(np.asarray(m.reference_load) - (np.asarray(m.pv_available) if m.has_pv else 0.0) for m in members)

    hours = np.arange(len(cap))
    nets = np.array([net_demand(s.members) for s in scenarios])
    base = net_demand(base_members)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1.6, 1]})
    ax1.fill_between(hours, nets.min(axis=0), nets.max(axis=0), step="mid", color="tab:blue", alpha=0.25,
                     label=f"Range over the {len(scenarios)} scenarios")
    for k, n in enumerate(nets):
        ax1.step(hours, n, where="mid", color="tab:blue", linewidth=0.7, alpha=0.6,
                 label="One scenario" if k == 0 else None)
    ax1.step(hours, base, where="mid", color="black", linewidth=1.8, label=r"Base case ($\theta = 1$)")
    ax1.step(hours, cap, where="mid", color="tab:red", linestyle="--", linewidth=1.5, label="Import cap")
    ax1.axhline(0, color="black", linewidth=0.6)
    ax1.set(xlabel="hour", ylabel="kW", xticks=hours[::2],
            title="(a) Net reference demand of the community (load - available PV)")
    ax1.legend(fontsize=8, loc="upper left")
    ax1.grid(alpha=0.3)

    s = np.arange(len(in_model_cost))
    ax2.bar(s, in_model_cost, 0.6, color="tab:blue", alpha=0.6, label="SDP, in the model")
    ax2.plot(s, evaluator_cost, "D", color="black", markersize=6, label="Same prices, evaluator")
    ax2.axhline(dp_cost, color="tab:gray", linestyle="--", linewidth=1.5, label="DP, expected parameters")
    diff = float(np.max(np.abs(np.asarray(evaluator_cost) - np.asarray(in_model_cost))))
    lo = min(np.min(in_model_cost), np.min(evaluator_cost), dp_cost)
    hi = max(np.max(in_model_cost), np.max(evaluator_cost), dp_cost)
    pad = 0.15 * (hi - lo) + 1.0
    ax2.set(ylim=(lo - pad, hi + pad), xticks=s, xlabel="in-sample scenario", ylabel="community cost (DKK/day)",
            title=f"(b) Cost per scenario (max difference {diff:.3f} DKK)")
    ax2.legend(fontsize=8, loc="lower left")
    ax2.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    path = output_dir / f"{tag}.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_q3_uncertainty_impact(ev, predicted_cost: float, predicted_surplus: float, output_dir: Path | str,
                               label: str = "DP", tag: str = "q3_uncertainty_impact") -> Path:
    """Question 3, step 1: a deterministic price vector replayed on out-of-sample scenarios.
    x: the community's daily reference energy in the scenario minus the base case (kWh/day), i.e. how much more the
    members want than declared. Top: realized community cost; bottom: the manager's surplus (payments - meter bill).
    Colour: cap exceedance (kWh/day). Dashed: what the deterministic model predicts (expected parameters)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dE, exc = ev["dE"].to_numpy(), ev["exceedance_kWh"].to_numpy()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7.5), sharex=True)
    vmax = max(float(exc.max()), 1e-3)
    for ax, col, pred, name in ((ax1, "cost", predicted_cost, "community cost"),
                                (ax2, "surplus", predicted_surplus, "manager surplus")):
        y = ev[col].to_numpy()
        sc = ax.scatter(dE, y, c=exc, cmap="Reds", vmin=0, vmax=vmax, edgecolor="black", linewidth=0.4, s=36, zorder=3)
        ax.axhline(pred, color="tab:blue", linestyle="--", linewidth=1.5,
                   label=f"{label} prediction (expected parameters): {pred:.1f} DKK")
        ax.axhline(y.mean(), color="black", linestyle=":", linewidth=1.5,
                   label=f"Realized mean over {len(y)} scenarios: {y.mean():.1f} DKK")
        ax.set_ylabel(f"{name} (DKK/day)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="upper left" if col == "cost" else "lower left")
    ax1.set_title(f"(a) Realized community cost at the posted prices of {label}")
    ax2.set_title("(b) Manager surplus (payments - meter bill); < 0 is a budget deficit")
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_xlabel("community reference energy minus base case (kWh/day): members want more than declared ->")
    for ax in (ax1, ax2):
        ax.axvline(0, color="gray", linewidth=0.8)
    fig.colorbar(sc, ax=(ax1, ax2), pad=0.02, label="cap exceedance (kWh/day)")
    path = output_dir / f"{tag}.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path
