"""Question 3: numerical experiments.

    python -m src.q3_experiments --step1               # DP's prices replayed on the out-of-sample scenarios (figure 1)
    python -m src.q3_experiments --runs                # solve every model of RUNS (about 4 h), skipping finished ones
    python -m src.q3_experiments --evaluate            # every finished run on the out-of-sample set + comparison table
    python -m src.q3_experiments --runs --evaluate     # both (the overnight command)
    python -m src.q3_experiments --list                # the runs and whether they are finished

Seeds (report them): in-sample scenarios seed 1 (N = 5, 10; nested: the first 5 of N = 10 are those of N = 5),
out-of-sample scenarios seed 1000 (100 scenarios), the same out-of-sample set for every model (common random numbers).
IR benchmark: model R per scenario. Price regularization towards C's internal prices lambda^C (weight 1e-3).

Step 1 (suggested step of the assignment): how do different draws of the profiles change the members' response at the
prices of the deterministic model DP (expected parameters), and in turn the manager's cost, budget and the members'
individual rationality? Writes to results/<community>/q3/:
    step1_dp_oos_scenarios.csv   one row per out-of-sample scenario (cost components, surplus, exceedance)
    step1_dp_oos_members.csv     one row per (scenario, member): cost, benchmark C^R_is, IR violation
    step1_dp_oos_metrics.csv     distribution metrics (mean, quantiles, CVaR, frequencies)
    q3_uncertainty_impact.png    figure 1

Runs (--runs): each one is solved once and saved to results/<community>/q3/runs/<name>/ (prices, in-sample scenario and
member tables, meta.json with status, gap, bound, time and size); a run whose folder is complete is skipped, so the
command can be restarted after an interruption. Each run is warm-started from the prices of the run it names in
"start" (see StochasticDynamicPricingModel.warm_start); the chain makes every start satisfy the run's requirements.

Evaluation (--evaluate): every finished run's prices on the 100 out-of-sample scenarios, written next to the run
(oos_scenarios.csv, oos_members.csv), and q3_comparison.csv / .tex: in-sample (targeted) and out-of-sample (achieved)
metrics side by side for every model.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import scenario_data as sd
from .data_loader import load_community
from .model import CommunityContractCommunityModel, DynamicPricingCommunityModel, internal_price_of_C
from .plotting import plot_q3_uncertainty_impact
from .stochastic import StochasticDynamicPricingModel, benchmark_costs, evaluate_prices, risk_metrics

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
SEED_IN, SEED_OUT, N_OUT = 1, 1000, 100

# name -> settings. n = 0: the expected scenario only (the deterministic models). Order matters: a run starts from
# the prices of an earlier one. box = how far the start may move the prices (DKK/kWh) to meet the run's requirements.
RUNS: dict[str, dict] = {
    "DP_BB": dict(label=r"$\mathcal{DP}^{BB}$", n=0, ir=False, time_limit=600),
    "DP_BB_IR": dict(label=r"$\mathcal{DP}^{BB,IR}$", n=0, ir=True, time_limit=1200, start="DP_BB"),
    "SDPx_N5": dict(label=r"$\mathcal{SDP}^{x}$, N=5", n=5, time_limit=1800, start="DP_BB_IR", box=0.05,
                    start_time_limit=300),
    "SDPcvar_N5": dict(label=r"$\mathcal{SDP}^{CVaR}$, N=5", n=5, objective="cvar", cvar_alpha=0.2, cvar_weight=0.5,
                       time_limit=1800, start="SDPx_N5", start_core=True),
    "SDPeps0_N5": dict(label=r"$\mathcal{SDP}^{\varepsilon=0}$, N=5", n=5, constraints="chance", epsilon=0.0,
                       time_limit=1800, start="DP"),
    "SDPeps20_N5": dict(label=r"$\mathcal{SDP}^{\varepsilon=0.2}$, N=5", n=5, constraints="chance", epsilon=0.2,
                        time_limit=1800, start="SDPeps0_N5", start_core=True),
    "SDPeps40_N5": dict(label=r"$\mathcal{SDP}^{\varepsilon=0.4}$, N=5", n=5, constraints="chance", epsilon=0.4,
                        time_limit=1800, start="SDPeps20_N5", start_core=True),
    "SDPx_N10": dict(label=r"$\mathcal{SDP}^{x}$, N=10", n=10, time_limit=1800, start="SDPx_N5", box=0.1,
                     start_time_limit=900),
}
# start: the run whose prices start this one ("DP": the cap-only DP, whose surplus of about +150 DKK meets budget
# balance >= 0 in every scenario); start_core: start from that run's whole solution (x, flows, multipliers), which
# needs the same in-sample scenarios and a run whose solution is feasible here (a stricter or equally strict one).
MODEL_KEYS = ("objective", "cvar_alpha", "cvar_weight", "constraints", "epsilon")


# ----------------------------------------------------------------------------------------------------------------------
# Step 1
# ----------------------------------------------------------------------------------------------------------------------
def deterministic_dp(com, time_limit: float | None = None):
    """DP on the expected parameters, regularized towards C's internal prices (as in Question 2)."""
    lam = c_prices(com)
    r = DynamicPricingCommunityModel(com, method="SOS1", reg_weight=1e-3, reg_reference=lam,
                                     time_limit=time_limit).build().solve()
    return r, lam


def c_prices(com) -> list[float]:
    C = CommunityContractCommunityModel(com).build()
    C.solve()
    return internal_price_of_C(C)


def step1(com, out: Path, n_out: int = N_OUT, seed_out: int = SEED_OUT) -> pd.DataFrame:
    r, _ = deterministic_dp(com)
    x = r.hourly["posted_price"].to_numpy()
    oos = sd.draw_scenarios(com, n=n_out, seed=seed_out)
    sd.check_scenarios(com, oos)
    bench = benchmark_costs(com, oos)
    ev, mem = evaluate_prices(com, x, oos, benchmark=bench, label="DP")
    ev["dE"] = ev["reference_energy_kWh"] - sum(m.reference_load.sum() for m in com.members)
    metrics = pd.DataFrame([{"model": "DP", "predicted_cost": r.meta["welfare_cost"],
                             "predicted_surplus": r.meta["manager_surplus"], **risk_metrics(ev, mem)}])
    ev.to_csv(out / "step1_dp_oos_scenarios.csv", index=False)
    mem.to_csv(out / "step1_dp_oos_members.csv", index=False)
    metrics.to_csv(out / "step1_dp_oos_metrics.csv", index=False)
    fig = plot_q3_uncertainty_impact(ev, r.meta["welfare_cost"], r.meta["manager_surplus"], out)
    print(metrics.T.round(3).to_string(header=False))
    print(f"figure: {fig}")
    return metrics


# ----------------------------------------------------------------------------------------------------------------------
# Runs
# ----------------------------------------------------------------------------------------------------------------------
def in_sample(com, n: int, seed: int = SEED_IN) -> list[sd.Scenario]:
    return [sd.expected_scenario(com)] if n == 0 else sd.draw_scenarios(com, n=n, seed=seed)


def run_done(folder: Path) -> bool:
    return (folder / "meta.json").exists() and (folder / "prices.csv").exists()


def load_prices(folder: Path) -> np.ndarray:
    return pd.read_csv(folder / "prices.csv")["price"].to_numpy()


def solve_run(com, name: str, cfg: dict, runs_dir: Path, lam, scale: float = 1.0) -> None:
    folder = runs_dir / name
    scen = in_sample(com, cfg["n"])
    bench = benchmark_costs(com, scen) if cfg.get("ir", True) else None
    start, core = None, None
    if cfg.get("start") == "DP":
        r_dp, _ = deterministic_dp(com)
        start = r_dp.hourly["posted_price"].to_numpy()
    elif cfg.get("start"):
        start_dir = runs_dir / cfg["start"]
        if not run_done(start_dir):
            raise RuntimeError(f"{name} starts from {cfg['start']}, which has not been solved")
        start = load_prices(start_dir)
        if cfg.get("start_core"):
            core = np.load(start_dir / "core_solution.npy")
    tl = cfg["time_limit"] * scale
    model = StochasticDynamicPricingModel(
        com, scen, name=name, budget_balance=True, ir_benchmark=bench, reg_reference=lam, time_limit=tl,
        mip_focus=1, start_prices=start, start_box=cfg.get("box", 0.0),
        start_time_limit=cfg.get("start_time_limit", 120) * min(scale, 1.0), start_solution=core,
        **{k: cfg[k] for k in MODEL_KEYS if k in cfg}).build()
    folder.mkdir(parents=True, exist_ok=True)
    model.m.Params.OutputFlag = 1                      # log to the file only (verbose=False turns logging off)
    model.m.Params.LogToConsole = 0
    model.m.Params.LogFile = str(folder / "gurobi.log")
    t0 = time.time()
    r = model.solve()
    pd.DataFrame({"hour": list(model.T), "price": r.prices}).to_csv(folder / "prices.csv", index=False)
    np.save(folder / "core_solution.npy", model.core_solution())
    r.scenarios.to_csv(folder / "in_sample_scenarios.csv", index=False)
    r.members.to_csv(folder / "in_sample_members.csv", index=False)
    meta = {"name": name, "label": cfg["label"], "config": cfg, "status": r.status, "objective": r.objective,
            "wall_time_s": time.time() - t0, "seed_in": SEED_IN, **r.meta}
    (folder / "meta.json").write_text(json.dumps(meta, indent=1, default=lambda o: o.tolist()
                                                 if hasattr(o, "tolist") else str(o)), encoding="utf-8")
    ws = r.meta.get("warm_start") or {}
    print(f"[{name}] {r.summary()} | start found {ws.get('found')}, used {ws.get('start_used')}", flush=True)


def run_all(com, out: Path, only=None, force: bool = False, scale: float = 1.0) -> None:
    runs_dir = out / "runs"
    lam = c_prices(com)
    for name, cfg in RUNS.items():
        if only and name not in only:
            continue
        if run_done(runs_dir / name) and not force:
            print(f"[{name}] already solved, skipped", flush=True)
            continue
        print(f"[{name}] solving (time limit {cfg['time_limit'] * scale:.0f} s) ...", flush=True)
        try:
            solve_run(com, name, cfg, runs_dir, lam, scale)
        except Exception as err:                       # keep going: later runs may not depend on this one
            print(f"[{name}] FAILED: {err}", flush=True)


# ----------------------------------------------------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------------------------------------------------
def evaluate_all(com, out: Path, n_out: int = N_OUT, seed_out: int = SEED_OUT) -> pd.DataFrame:
    runs_dir = out / "runs"
    oos = sd.draw_scenarios(com, n=n_out, seed=seed_out)
    bench = benchmark_costs(com, oos)
    rows = []
    for name in RUNS:
        folder = runs_dir / name
        if not run_done(folder):
            continue
        meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
        ev, mem = evaluate_prices(com, load_prices(folder), oos, benchmark=bench, label=name)
        ev.to_csv(folder / "oos_scenarios.csv", index=False)
        mem.to_csv(folder / "oos_members.csv", index=False)
        ins = risk_metrics(pd.read_csv(folder / "in_sample_scenarios.csv"), pd.read_csv(folder / "in_sample_members.csv"))
        oo = risk_metrics(ev, mem)
        size = meta.get("size", {})
        rows.append({"model": name, "label": meta["label"], "N": meta["config"]["n"], "status": meta["status"],
                     "gap": meta["mip_gap"], "time_s": meta["runtime_s"], "binaries": size.get("binaries"),
                     "sos": size.get("sos"),
                     **{f"in_{k}": v for k, v in ins.items()}, **{f"oos_{k}": v for k, v in oo.items()}})
        print(f"[{name}] evaluated: E[cost] in {ins['E_cost']:.2f} -> out {oo['E_cost']:.2f} DKK, "
              f"P(deficit) in {ins['P_deficit']:.2f} -> out {oo['P_deficit']:.2f}", flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(out / "q3_comparison.csv", index=False)
    write_comparison_tex(table, out / "q3_comparison.tex")
    print(f"comparison: {out / 'q3_comparison.csv'}")
    return table


def write_comparison_tex(table: pd.DataFrame, path: Path) -> None:
    """In-sample (targeted) and out-of-sample (achieved) metrics side by side, one row per model."""
    cols = [("E_cost", "E[cost]", 1), ("CVaR20_cost", r"CVaR$_{20}$", 1), ("P_deficit", "P(deficit)", 2),
            ("E_deficit", "E[deficit]", 2), ("P_ir_violation", "P(IR viol.)", 2), ("E_ir_violation", "E[IR viol.]", 2),
            ("P_exceedance", "P(exceed.)", 2)]
    head = " & ".join([r"Model", "gap", "time (s)"] + [f"{h} in / out" for _, h, _ in cols])
    lines = [r"\begin{table}[ht]", r"\centering", r"\scriptsize",
             r"\caption{Question 3: in-sample (targeted) / out-of-sample (achieved, 100 scenarios, seed 1000) "
             r"metrics. Costs and violations in DKK/day.}", r"\label{tab:q3_comparison}",
             r"\begin{tabular}{l" + "r" * (2 + len(cols)) + "}", r"\toprule", head + r" \\", r"\midrule"]
    for _, r in table.iterrows():
        cells = [r["label"], f"{r['gap'] * 100:.1f}\\%", f"{r['time_s']:.0f}"]
        for key, _, d in cols:
            a, b = r.get(f"in_{key}", np.nan), r.get(f"oos_{key}", np.nan)
            cells.append("--" if pd.isna(a) else f"{a:.{d}f} / {b:.{d}f}")
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--community", default="base")
    ap.add_argument("--step1", action="store_true", help="DP's prices on the out-of-sample scenarios (figure 1)")
    ap.add_argument("--runs", action="store_true", help="solve the models of RUNS that are not solved yet")
    ap.add_argument("--evaluate", action="store_true", help="evaluate every solved run out of sample")
    ap.add_argument("--list", action="store_true", help="list the runs and their state")
    ap.add_argument("--only", nargs="*", default=None, help="restrict --runs to these run names")
    ap.add_argument("--force", action="store_true", help="re-solve runs that are already solved")
    ap.add_argument("--time-scale", type=float, default=1.0,
                    help="multiply every time limit (e.g. 0.05 for a quick test of the whole chain)")
    ap.add_argument("--out", default=None, help="output folder (default results/<community>/q3)")
    ap.add_argument("--n-out", type=int, default=N_OUT)
    ap.add_argument("--seed-out", type=int, default=SEED_OUT)
    args = ap.parse_args()
    com = load_community(args.community)
    out = Path(args.out) if args.out else RESULTS_DIR / args.community / "q3"
    out.mkdir(parents=True, exist_ok=True)
    if args.list:
        for name, cfg in RUNS.items():
            state = "solved" if run_done(out / "runs" / name) else "not solved"
            print(f"  {name:12s} {state:10s} time limit {cfg['time_limit']:5d} s, start from {cfg.get('start', '-')}")
        total = sum(c["time_limit"] for c in RUNS.values())
        print(f"  total time limits: {total / 3600:.1f} h (plus warm starts)")
    if args.step1:
        step1(com, out, args.n_out, args.seed_out)
    if args.runs:
        t0 = time.time()
        run_all(com, out, args.only, args.force, args.time_scale)
        print(f"runs finished in {(time.time() - t0) / 3600:.2f} h", flush=True)
    if args.evaluate:
        evaluate_all(com, out, args.n_out, args.seed_out)


if __name__ == "__main__":
    main()
