"""Question 3: figures for uncertainty analysis, validation, and saved stochastic runs.

    python -m src.q3_plotting                                      # results/base/q3/runs -> results/base/q3/
    python -m src.q3_plotting --runs-dir results/base/q3/runs_benchR --tag benchR

Figure 2  q3_models.png      Do the models deliver what they promise? Realized cost, manager surplus and IR violation
                             on the out-of-sample scenarios (box: 5-95 % whiskers, line: median, o: mean) against
                             what each model expects in-sample (diamond).
Figure 3  q3_epsilon.png     The chance constraint: violation probability targeted (epsilon), reached in-sample and
                             achieved out-of-sample; what the protected budget costs in IR violation and in cost.
Figure 4  q3_scenarios_N.png How many scenarios? In-sample vs out-of-sample cost (optimism gap), budget and IR
                             violations, and model size / optimality gap against N.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .q3_stochastic import risk_metrics

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
MODELS_FIG2 = ["DP_BB_IR", "SDPx_N5", "SDPcvar_N5", "SDPeps0_N5", "SDPeps20_N5", "SDPeps40_N5"]
EPS_RUNS = {0.0: "SDPeps0_N5", 0.2: "SDPeps20_N5", 0.4: "SDPeps40_N5"}
COLOURS = {"DP_BB_IR": "tab:gray", "SDPx_N5": "tab:blue", "SDPcvar_N5": "tab:purple", "SDPeps0_N5": "tab:green",
           "SDPeps20_N5": "tab:olive", "SDPeps40_N5": "tab:orange", "SDPx_N10": "tab:cyan"}
IN_STYLE = dict(marker="D", color="black", markersize=7, linestyle="none", zorder=5)


def load_run(folder: Path) -> dict | None:
    """Everything a figure needs from one run folder, or None if it has not been evaluated out of sample."""
    if not (folder / "oos_scenarios.csv").exists():
        return None
    meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    run = {"meta": meta, "label": meta["label"], "N": meta["config"]["n"]}
    for part in ("in_sample", "oos"):
        scen = pd.read_csv(folder / f"{part}_scenarios.csv")
        mem = pd.read_csv(folder / f"{part}_members.csv")
        ir = mem.groupby("scenario")["ir_violation"].sum().reindex(scen["scenario"]).to_numpy()
        scen = scen.assign(ir_violation=ir)
        run[part] = scen
        run[f"{part}_metrics"] = risk_metrics(scen, mem)
    return run


def load_runs(runs_dir: Path) -> dict[str, dict]:
    runs = {}
    for folder in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        r = load_run(folder)
        if r is not None:
            runs[folder.name] = r
    return runs


def _mean(df: pd.DataFrame, col: str) -> float:
    w = df["probability"].to_numpy() / df["probability"].sum()
    return float(w @ df[col].to_numpy())


# ----------------------------------------------------------------------------------------------------------------------
# Figure 2
# ----------------------------------------------------------------------------------------------------------------------
def fig_models(runs: dict, out: Path, tag: str = "") -> Path | None:
    names = [n for n in MODELS_FIG2 if n in runs]
    if not names:
        return None
    panels = [("cost", "(a) Community cost (DKK/day)", None),
              ("surplus", "(b) Manager surplus (DKK/day); < 0 = budget deficit", 0.0),
              ("ir_violation", "(c) IR violation, summed over members (DKK/day)", 0.0)]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    pos = np.arange(len(names))
    for ax, (col, title, ref) in zip(axes, panels):
        data = [runs[n]["oos"][col].to_numpy() for n in names]
        bp = ax.boxplot(data, positions=pos, widths=0.55, whis=(5, 95), showmeans=True, patch_artist=True,
                        showfliers=True, flierprops=dict(marker=".", markersize=3, alpha=0.5),
                        meanprops=dict(marker="o", markerfacecolor="white", markeredgecolor="black", markersize=5),
                        medianprops=dict(color="black"))
        for patch, n in zip(bp["boxes"], names):
            patch.set(facecolor=COLOURS.get(n, "tab:blue"), alpha=0.45)
        ax.plot(pos, [_mean(runs[n]["in_sample"], col) for n in names], **IN_STYLE)
        if ref is not None:
            ax.axhline(ref, color="black", linewidth=0.8)
        if col == "surplus":
            lo = min(float(d.min()) for d in data)
            ax.axhspan(min(lo * 1.1, -1.0), 0, color="tab:red", alpha=0.06)
        ax.set_xticks(pos)
        ax.set_xticklabels([runs[n]["label"].replace(", N=5", "") for n in names], rotation=30, ha="right")
        ax.set_title(title, fontsize=11)
        ax.grid(alpha=0.3, axis="y")
    handles = [plt.Line2D([], [], **IN_STYLE, label="In-sample expectation (what the model promises)"),
               plt.Line2D([], [], marker="o", markerfacecolor="white", markeredgecolor="black", linestyle="none",
                          label="Out-of-sample mean (100 scenarios, seed 1000)"),
               plt.Line2D([], [], color="black", label="Out-of-sample median; box 25-75 %, whiskers 5-95 %")]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=9, frameon=False)
    fig.suptitle("Promised (in-sample, N = 5) and realized (out-of-sample) performance of the posted prices", y=0.99)
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    path = out / f"q3_models{'_' + tag if tag else ''}.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


# ----------------------------------------------------------------------------------------------------------------------
# Figure 3
# ----------------------------------------------------------------------------------------------------------------------
def fig_epsilon(runs: dict, out: Path, tag: str = "") -> Path | None:
    eps = [e for e, n in EPS_RUNS.items() if n in runs]
    if len(eps) < 2:
        return None
    R = [runs[EPS_RUNS[e]] for e in eps]
    x = np.array(eps)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4.8))

    ax1.plot(x, x, "-", color="tab:gray", linewidth=7, alpha=0.3, zorder=1,
             label=r"Target: at most $\varepsilon$ (in-sample lies on it)")
    ax1.plot(x, [r["in_sample_metrics"]["P_deficit"] for r in R], "s-", color="tab:blue", label="Reached in-sample")
    ax1.plot(x, [r["oos_metrics"]["P_deficit"] for r in R], "o-", color="tab:red", label="Achieved out-of-sample")
    ax1.set(xlabel=r"$\varepsilon$ (allowed share of violating scenarios)", ylabel="share of scenarios with a deficit",
            ylim=(-0.03, 1.0), xticks=x, title="(a) Budget deficits: targeted vs achieved")

    ax2.plot(x, [r["oos_metrics"]["E_deficit"] for r in R], "o-", color="tab:red", label="Mean budget deficit")
    ax2.plot(x, [r["oos_metrics"]["E_ir_violation"] for r in R], "o-", color="tab:green",
             label="Mean IR violation (summed over members)")
    ax2.set(xlabel=r"$\varepsilon$", ylabel="DKK/day (out-of-sample)", xticks=x,
            title="(b) Who carries the risk: the manager or the members")

    ax3.plot(x, [r["in_sample_metrics"]["E_cost"] for r in R], "s-", color="tab:blue", label="E[cost], in-sample")
    ax3.plot(x, [r["oos_metrics"]["E_cost"] for r in R], "o-", color="tab:red", label="E[cost], out-of-sample")
    ax3.plot(x, [r["oos_metrics"]["CVaR20_cost"] for r in R], "o--", color="tab:red", alpha=0.6,
             label=r"CVaR$_{20\%}$, out-of-sample")
    ax3.set(xlabel=r"$\varepsilon$", ylabel="community cost (DKK/day)", xticks=x, title="(c) Cost")
    for ax in (ax1, ax2, ax3):
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(r"Chance-constrained model $\mathcal{SDP}^{\varepsilon}$ (N = 5): budget balance in all but a share "
                 r"$\varepsilon$ of the scenarios", y=1.0)
    fig.tight_layout()
    path = out / f"q3_epsilon{'_' + tag if tag else ''}.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


# ----------------------------------------------------------------------------------------------------------------------
# Figure 4
# ----------------------------------------------------------------------------------------------------------------------
def fig_scenarios_N(runs: dict, out: Path, tag: str = "") -> Path | None:
    sdpx = sorted((r["N"], name) for name, r in runs.items() if name.startswith("SDPx_N"))
    if len(sdpx) < 2:
        return None
    Ns = [n for n, _ in sdpx]
    R = [runs[name] for _, name in sdpx]
    det = runs.get("DP_BB_IR")
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4.8))

    ax1.plot(Ns, [r["in_sample_metrics"]["E_cost"] for r in R], "s-", color="tab:blue", label="Promised (in-sample)")
    ax1.plot(Ns, [r["oos_metrics"]["E_cost"] for r in R], "o-", color="tab:red", label="Achieved (out-of-sample)")
    for n, r in zip(Ns, R):
        a, b = r["in_sample_metrics"]["E_cost"], r["oos_metrics"]["E_cost"]
        ax1.vlines(n, min(a, b), max(a, b), color="black", linewidth=0.8, linestyle=":")
        ax1.annotate(f"{b - a:+.1f}", (n, (a + b) / 2), xytext=(5, 0), textcoords="offset points", fontsize=9,
                     va="center")
    if det is not None:
        ax1.axhline(det["oos_metrics"]["E_cost"], color="tab:gray", linestyle=":",
                    label=r"$\mathcal{DP}^{BB,IR}$, achieved")
    ax1.set(xlabel="in-sample scenarios N", ylabel="E[community cost] (DKK/day)", xticks=Ns,
            title="(a) Optimism gap: achieved - promised")

    ax2.plot(Ns, [r["oos_metrics"]["E_deficit"] for r in R], "o-", color="tab:red", label="Mean budget deficit (DKK)")
    ax2.plot(Ns, [r["oos_metrics"]["E_ir_violation"] for r in R], "o-", color="tab:green",
             label="Mean IR violation (DKK)")
    ax2b = ax2.twinx()
    ax2b.plot(Ns, [r["oos_metrics"]["P_deficit"] for r in R], "s--", color="tab:red", alpha=0.5,
              label="Share of scenarios with a deficit")
    ax2b.set(ylim=(0, 1), ylabel="share of scenarios")
    ax2.set(xlabel="in-sample scenarios N", ylabel="DKK/day (out-of-sample)", xticks=Ns,
            title="(b) Violations out-of-sample")
    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = ax2b.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, fontsize=8)

    size = [r["meta"]["size"]["complementarity_pairs"] for r in R]
    ax3.bar([str(n) for n in Ns], size, color="tab:blue", alpha=0.5)
    for k, r in enumerate(R):
        ax3.text(k, size[k], f"gap {100 * r['meta']['mip_gap']:.0f} %\n{r['meta']['runtime_s']:.0f} s",
                 ha="center", va="bottom", fontsize=8)
    ax3.set(xlabel="in-sample scenarios N", ylabel="complementarity pairs (SOS1)", ylim=(0, 1.25 * max(size)),
            title="(c) Model size and optimality gap at the time limit")
    for ax in (ax1, ax2, ax3):
        ax.grid(alpha=0.3)
    ax1.legend(fontsize=8)
    fig.suptitle(r"$\mathcal{SDP}^{x}$ against the number of in-sample scenarios (nested draws, seed 1; "
                 r"same 100 out-of-sample scenarios)", y=1.0)
    fig.tight_layout()
    path = out / f"q3_scenarios_N{'_' + tag if tag else ''}.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_q3_validation(scenarios, base_members, cap, in_model_cost, evaluator_cost, dp_cost: float,
                       output_dir: Path | str, tag: str = "q3_validation") -> Path:
    """Plot scenario demand variation and the model/evaluator cost validation."""
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
    """Plot realized cost and manager surplus against out-of-sample uncertainty."""
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


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--community", default="base")
    ap.add_argument("--runs-dir", default=None, help="default results/<community>/q3/runs")
    ap.add_argument("--out", default=None, help="default results/<community>/q3")
    ap.add_argument("--tag", default="", help="suffix of the file names, e.g. benchR")
    args = ap.parse_args(argv)
    base = RESULTS_DIR / args.community / "q3"
    runs_dir = Path(args.runs_dir) if args.runs_dir else base / "runs"
    out = Path(args.out) if args.out else base
    out.mkdir(parents=True, exist_ok=True)
    runs = load_runs(runs_dir)
    print(f"{len(runs)} evaluated runs in {runs_dir}: {', '.join(runs)}")
    for f in (fig_models, fig_epsilon, fig_scenarios_N):
        path = f(runs, out, args.tag)
        print(f"  {f.__name__}: {path or 'skipped (runs missing)'}")


if __name__ == "__main__":
    main()
