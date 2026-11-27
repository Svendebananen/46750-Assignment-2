"""Plots for the dynamic-pricing experiments in Question 2."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

if TYPE_CHECKING:
    from .model import Results


# Question 2: dynamic pricing
# ----------------------------------------------------------------------------------------------------------------------
_RUN_COLOURS = ["tab:blue", "tab:orange", "tab:green"]          # one colour per DP step, the default matplotlib cycle


def _cap_binding_hours(hourly, cap: np.ndarray, tol: float = 1e-3) -> np.ndarray:
    """True in the hours in which the community import sits at the cap."""
    imp = hourly["import"].to_numpy()
    return np.array([abs(i - c) <= tol * max(1.0, c) for i, c in zip(imp, cap)])


def _meter_prices(community):
    """p_t, and the community's marginal import and export prices at the meter."""
    beta = community.contract.tariff_discount_ratio
    p = np.asarray(community.energy_price)
    return p, p + (1 - beta) * community.import_tariff, p - (1 - beta) * community.export_tariff


def plot_dp_prices(result: "Results", lambda_C: list[float], community, output_dir: Path | str, tag: str = "") -> Path:
    """Top: posted price x_t, C's internal price and the meter prices. Bottom: import and export against the cap."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    hourly = result.hourly
    hours = np.asarray(hourly.index)
    p, imp_price, exp_price = _meter_prices(community)
    cap = np.asarray(community.contract.capacity_limit_kW)
    binding = _cap_binding_hours(hourly, cap)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    ax1.plot(hours, hourly["posted_price"], label="Posted price x_t (DP)", linewidth=2)
    ax1.plot(hours, lambda_C, "--", label="Internal price of C", linewidth=2)
    ax1.plot(hours, p, color="tab:purple", label="Day-ahead price", linewidth=2)
    ax1.fill_between(hours, exp_price, imp_price, color="tab:purple", alpha=0.15, label="Meter export-import price")
    if binding.any():
        ax1.fill_between(hours, 0, 1, where=binding, transform=ax1.get_xaxis_transform(), color="tab:gray",
                         alpha=0.15, step="mid", label="Import at the cap")
    ax1.set(title="DP: posted prices and the internal prices of C", ylabel="Price (DKK/kWh)")
    ax1.grid(True, alpha=0.25)
    ax1.legend(ncol=2)

    ax2.plot(hours, hourly["import"], label="Grid import", linewidth=2)
    ax2.plot(hours, cap, "--", label="Contract import cap", linewidth=2)
    ax2.plot(hours, hourly["export"], label="Grid export", linewidth=2)
    if binding.any():
        ax2.fill_between(hours, 0, 1, where=binding, transform=ax2.get_xaxis_transform(), color="tab:gray",
                         alpha=0.15, step="mid")
    ax2.set(title="DP: grid import and capacity limit", xlabel="Hour of day", ylabel="Energy (kWh)", xticks=hours)
    ax2.grid(True, alpha=0.25)
    ax2.legend()
    fig.tight_layout()
    path = output_dir / f"{result.question}{'_' + tag if tag else ''}_prices.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_dp_compare_prices(runs: dict, lambda_C: list[float], community, output_dir: Path | str,
                           tag: str = "dynamic_compare") -> Path:
    """The DP steps on one hour axis: posted prices (top) and grid import against the cap (bottom)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    p, imp_price, exp_price = _meter_prices(community)
    cap = np.asarray(community.contract.capacity_limit_kW)
    first = next(iter(runs.values()))
    hours = np.asarray(first.hourly.index)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for k, (label, r) in enumerate(runs.items()):
        ax1.plot(hours, r.hourly["posted_price"], color=_RUN_COLOURS[k], label=f"Posted price: {label}", linewidth=2)
        ax2.plot(hours, r.hourly["import"], color=_RUN_COLOURS[k], label=f"Grid import: {label}", linewidth=2)
    ax1.plot(hours, lambda_C, "--", color="black", label="Internal price of C", linewidth=1.5)
    ax1.plot(hours, p, color="tab:purple", label="Day-ahead price", linewidth=2)
    ax1.fill_between(hours, exp_price, imp_price, color="tab:purple", alpha=0.15, label="Meter export-import price")
    ax2.plot(hours, cap, "--", color="black", label="Contract import cap", linewidth=2)
    top = 1.25 * max(np.max(imp_price), np.max(lambda_C))                # a price at the bound X is written, not drawn
    for k, (label, r) in enumerate(runs.items()):
        for t, v in zip(hours, r.hourly["posted_price"]):
            if v > top:
                ax1.annotate(f"{v:.1f}", (t, top), xytext=(0, -14), textcoords="offset points", ha="center",
                             fontsize=8, color=_RUN_COLOURS[k], arrowprops={"arrowstyle": "->", "color": _RUN_COLOURS[k]})
    ax1.set_ylim(top=top)
    ax1.set(title="DP: posted prices for each step", ylabel="Price (DKK/kWh)")
    ax2.set(title="DP: grid import and capacity limit for each step", xlabel="Hour of day",
            ylabel="Energy (kWh)", xticks=hours)
    for ax in (ax1, ax2):
        ax.grid(True, alpha=0.25)
        ax.legend()
    fig.tight_layout()
    path = output_dir / f"{first.question}_{tag}_prices.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path

_STEP_COLOURS = {"C (ref.)": "tab:gray", "C (dual prices)": "tab:gray",
                 "DP": "tab:blue", "DP^BB": "tab:orange", "DP^BB,IR": "tab:green"}
_COMPONENT_COLOURS = {"energy": "tab:blue", "tariffs": "tab:orange", "penalty": "tab:red", "PV + discomfort": "tab:green"}


def _tex(name: str) -> str:
    """DP^BB,IR -> DP$^{BB,IR}$ for titles and legends."""
    if "^" not in name:
        return name
    base, sup = name.split("^", 1)
    return f"{base}$^{{{sup}}}$"


def _min_span(ax, span: float = 0.1) -> None:
    """Keep a readable y-range when every value is (close to) zero."""
    lo, hi = ax.get_ylim()
    if hi - lo < span:
        ax.set_ylim(min(lo, -span / 2), max(hi, span / 2))


def plot_dp_cost_of_requirements(table, output_dir: Path | str, tag: str = "dynamic_q2") -> Path:
    """Left: what each step costs against C, split into components; the dot is the total.
    Right: the manager's surplus and the total loss of the members worse off than their benchmark."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    steps = table[table["model"] != "C (ref.)"]
    pos = np.arange(len(steps))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5))
    up, down = np.zeros(len(steps)), np.zeros(len(steps))
    for comp, colour in _COMPONENT_COLOURS.items():
        d = steps[f"d {comp}"].to_numpy()
        ax1.bar(pos, np.where(d > 0, d, 0), 0.6, bottom=up, color=colour, label=comp)
        ax1.bar(pos, np.where(d < 0, d, 0), 0.6, bottom=down, color=colour)
        up, down = up + np.where(d > 0, d, 0), down + np.where(d < 0, d, 0)
    total = steps["dW vs C"].to_numpy()
    ax1.plot(pos, total, "o", color="black", label="Welfare cost vs C (total)")
    for x, v in zip(pos, total):
        ax1.annotate(f"{v:+.2f}", (x, v), textcoords="offset points", xytext=(14, 0), va="center", fontsize=9)
    ax1.axhline(0, color="black", linewidth=0.8)
    _min_span(ax1)
    ax1.set(title="Cost of each requirement against C", ylabel="Difference to C (DKK/day)", xticks=pos)
    ax1.set_xticklabels([_tex(m) for m in steps["model"]])
    ax1.legend()

    pos2 = np.arange(len(table))
    w = 0.38
    ax2.bar(pos2 - w / 2, table["manager surplus"], w, color="tab:purple", label="Manager surplus")
    ax2.bar(pos2 + w / 2, table["loss of members worse off"], w, color="tab:brown", label="Loss of members worse off")
    for x, n, v in zip(pos2, table["members worse off"], table["loss of members worse off"]):
        ax2.annotate(f"{n} worse off", (x + w / 2, v), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=8)
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set(title="Where the money goes", ylabel="DKK/day", xticks=pos2)
    ax2.set_xticklabels([_tex(m) for m in table["model"]])
    ax2.legend()
    for ax in (ax1, ax2):
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path = output_dir / f"{tag}_cost_of_requirements.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_dp_benefits_by_member(tables: dict, benchmark: str, output_dir: Path | str, tag: str = "dynamic_q2") -> Path:
    """Each member's benefit against its benchmark (> 0 = better off), one bar per model, members grouped by assets.
    Bars below zero: individual rationality is violated (the blocking members)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    first = next(iter(tables.values()))
    ids, assets = first["member"].tolist(), first["assets"].tolist()
    pos = np.arange(len(ids))
    n = len(tables)
    width = 0.8 / n
    fig, ax = plt.subplots(figsize=(12, 6))
    for k, (name, mt) in enumerate(tables.items()):
        b = mt.set_index("member").loc[ids, "benefit"].to_numpy()
        x = pos + (k - (n - 1) / 2) * width
        ax.bar(x, b, width, color=_STEP_COLOURS.get(name, f"C{k}"),
               label=f"{_tex(name)}: {int((b < -1e-6).sum())} worse off, total {b.sum():+.1f} DKK")
    for j in range(1, len(ids)):
        if assets[j] != assets[j - 1]:
            ax.axvline(j - 0.5, color="black", linewidth=0.6, linestyle=":")
    for a in dict.fromkeys(assets):
        idx = [j for j, aj in enumerate(assets) if aj == a]
        ax.text(np.mean(idx), 1.01, a, transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=9)
    ax.axhline(0, color="black", linewidth=0.8)
    name = {"R": "R (no contract)", "I": "I (individual contract)", "best": "the better of R and I"}[benchmark]
    ax.set_title(f"Benefit per member against {name}", pad=22)
    ax.set(ylabel="Benefit (DKK/day)", xticks=pos)
    ax.set_xticklabels(ids, rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path = output_dir / f"{tag}_member_benefits.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_dp_cap_sweep(sweep, output_dir: Path | str, tag: str = "dynamic_q2") -> Path:
    """Sensitivity on the cap tightness f'. Top: cost of each step against C. Bottom: the rent DP collects and the
    compensation still needed with IR (number = members worse off). Hollow markers: optimality not proven."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    C = sweep[sweep["model"] == "C (ref.)"].set_index("f")["welfare cost"]
    for name in ("DP", "DP^BB", "DP^BB,IR"):
        s = sweep[sweep["model"] == name].set_index("f")
        if s.empty:
            continue
        d = s["welfare cost"] - C.loc[s.index]
        d = d.where(d.abs() > 1e-6, 0.0)
        ax1.plot(s.index, d, "-", color=_STEP_COLOURS[name], label=_tex(name), linewidth=2)
        proven = s["MIP gap"] <= 1e-4
        ax1.plot(s.index[proven], d[proven], "o", color=_STEP_COLOURS[name])
        ax1.plot(s.index[~proven], d[~proven], "o", mfc="white", color=_STEP_COLOURS[name])
    _min_span(ax1)
    ax1.set(title="Cap tightness: cost of each requirement against C", ylabel="Welfare cost - C (DKK/day)")
    dp = sweep[sweep["model"] == "DP"].set_index("f")
    ax2.plot(dp.index, dp["manager surplus"], "-o", color="tab:purple", label="Rent collected by DP (to give back)",
             linewidth=2)
    ir = sweep[sweep["model"] == "DP^BB,IR"].set_index("f")
    if not ir.empty:
        ax2.plot(ir.index, ir["sum zeta"], "-o", color=_STEP_COLOURS["DP^BB,IR"],
                 label="Compensation still needed with IR (sum of slacks)", linewidth=2)
        for f, z, n in zip(ir.index, ir["sum zeta"], ir["members worse off"]):
            ax2.annotate(f"{n}", (f, z), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)
    ax2.set(title="Rent to give back and compensation still needed (number = members worse off)",
            xlabel="Cap tightness f' (base 1.3)", ylabel="DKK/day", xticks=sorted(sweep["f"].unique()))
    for ax in (ax1, ax2):
        ax.axvline(1.3, color="black", linestyle=":", linewidth=1)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.grid(True, alpha=0.25)
        ax.legend()
    fig.tight_layout()
    path = output_dir / f"{tag}_cap_sweep.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path

def plot_dp_shift_heatmap(shifts: dict, assets: dict, cap, imports: dict, output_dir: Path | str,
                          base: str = "DP", compare: str | None = "DP^BB,IR", tag: str = "dynamic_q2") -> Path:
    """Shifted load per member and hour, L_it - L_it^ref (kWh), for two steps against the SAME reference (each
    member's preferred profile) and on the SAME colour scale, in the style of the Q1 heat maps (RdYlGn, centred on 0,
    values in the cells). Green = more load than preferred, red = less. Top: `base`. Bottom: `compare`.
    Black frames: hours in which `compare` imports above the cap. Dotted lines separate the asset groups."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    panels = [(shifts[base], f"{_tex(base)}: shifted load (kWh)")]
    if compare is not None and compare in shifts:
        panels.append((shifts[compare], f"{_tex(compare)}: shifted load (kWh)"))
    ids = list(panels[0][0].index)
    hours = np.arange(panels[0][0].shape[1])
    v = max(max(float(np.abs(d.to_numpy()).max()) for d, _ in panels), 1e-3)   # one symmetric scale for both
    fig, axes = plt.subplots(len(panels), 1, figsize=(13, 0.24 * len(ids) * len(panels) + 1.4 * len(panels)),
                             squeeze=False)
    for ax, (data, title) in zip(axes[:, 0], panels):
        z = data.loc[ids].to_numpy()
        im = ax.imshow(z, aspect="auto", cmap="RdYlGn", vmin=-v, vmax=v, interpolation="nearest")
        for (r, c), val in np.ndenumerate(z):                          # write the value in every shifted cell
            if abs(val) >= 0.05:
                ax.text(c, r, f"{val:.1f}", ha="center", va="center", fontsize=6, color="black")
        for j in range(1, len(ids)):                                  # separators between the asset groups
            if assets[ids[j]] != assets[ids[j - 1]]:
                ax.axhline(j - 0.5, color="black", linewidth=0.8, linestyle=":")
        ax.set_yticks(range(len(ids)))
        ax.set_yticklabels(ids, fontsize=8)
        ax.set_xticks(hours)
        ax.set(title=title, xlabel="hour", ylabel="member")
        fig.colorbar(im, ax=ax, pad=0.01)
    if compare is not None and compare in imports:                    # frame the hours where `compare` exceeds the cap
        over = np.asarray(imports[compare]) > np.asarray(cap) + 1e-6
        for ax in axes[:, 0]:
            for t in hours[over]:
                ax.add_patch(Rectangle((t - 0.5, -0.5), 1, len(ids), fill=False, edgecolor="black", linewidth=1.5))
    fig.tight_layout()
    path = output_dir / f"{tag}_member_shift.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


class DynamicPricingPlotter:
    """Create Question 2 figures in one output directory."""

    def __init__(self, output_dir: Path | str):
        self.output_dir = Path(output_dir)

    def prices(self, result, lambda_C, community, tag: str = "") -> Path:
        return plot_dp_prices(result, lambda_C, community, self.output_dir, tag)

    def compare_prices(self, runs, lambda_C, community, tag: str) -> Path:
        return plot_dp_compare_prices(runs, lambda_C, community, self.output_dir, tag)

    def cost_of_requirements(self, table, tag: str) -> Path:
        return plot_dp_cost_of_requirements(table, self.output_dir, tag)

    def benefits_by_member(self, tables, benchmark: str, tag: str) -> Path:
        return plot_dp_benefits_by_member(tables, benchmark, self.output_dir, tag)

    def cap_sweep(self, sweep, tag: str) -> Path:
        return plot_dp_cap_sweep(sweep, self.output_dir, tag)

    def shift_heatmap(self, shifts, assets, cap, imports, base: str = "DP",
                      compare: str | None = "DP^BB,IR", tag: str = "dynamic_q2") -> Path:
        return plot_dp_shift_heatmap(
            shifts, assets, cap, imports, self.output_dir, base, compare, tag
        )