"""Question 3: validation of the scenarios and of the stochastic pricing model SDP.

    python -m src.q3_validation                    # all checks, 300 s per MIP (about 15-20 min)
    python -m src.q3_validation --time-limit 60    # quicker (V5, V6 may stop at a gap)
    python -m src.q3_validation --only V1 V2 V3 V4 # a subset (about 1 min)

V1  scenarios: probabilities, theta inside [lo, hi] and averaging to the set mean, daily minimum = reference energy,
    indices recover the draws, nested sets (same seed, larger n -> same first draws)
V2  SDP on the expected scenario (theta = 1) reproduces DP (objective and prices)
V3  SDP on 3 identical copies of the expected scenario reproduces DP
V4  the fixed-price evaluator at DP's prices reproduces DP's community cost on the expected scenario
V5  SDP on N in-sample scenarios: complementarity and strong-duality residuals, and the evaluator replays the in-sample
    scenario costs at SDP's prices (evaluator <= in-model: optimistic tie-break, MIP may stop at a gap)
V6  CVaR with weight 0, and with alpha = 1 and weight 1, equals the risk-neutral objective

Output (results/<community>/q3/): q3_validation.csv and .tex (one row per check, for the appendix) and, when V5 runs,
q3_validation.png (the figure for the report).

The price regularization pulls towards C's internal prices lambda^C (as in Question 2): then DP without requirements
solves at the root node. Prices are warm-started from DP's.
"""
from __future__ import annotations

import argparse
import dataclasses
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import scenario_data as sd
from . import scenario_generation as sg
from .data_loader import load_community
from .model import CommunityContractCommunityModel, DynamicPricingCommunityModel, internal_price_of_C
from .plotting import plot_q3_validation
from .stochastic import StochasticDynamicPricingModel, evaluate_prices

TOL = 1e-6
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
ROWS: list[dict] = []


def check(name: str, test: str, expected: str, measured: str, tolerance: str, ok: bool, detail: str = "") -> bool:
    """Print one check and keep it as a row of the validation table."""
    ROWS.append({"check": name, "test": test, "expected": expected, "measured": measured, "tolerance": tolerance,
                 "result": "pass" if ok else "FAIL"})
    print(f"[{name}] {'PASS' if ok else 'FAIL'}  {test}: {measured} (expected {expected}, tol. {tolerance})"
          + (f"  | {detail}" if detail else ""), flush=True)
    return ok


def write_table(rows: list[dict], folder: Path) -> tuple[Path, Path]:
    df = pd.DataFrame(rows)
    csv = folder / "q3_validation.csv"
    df.to_csv(csv, index=False)
    esc = lambda v: str(v).replace("_", r"\_").replace("%", r"\%").replace("<=", r"$\le$").replace("theta", r"$\theta$")
    lines = [r"\begin{table}[ht]", r"\centering", r"\small",
             r"\caption{Validation of the scenarios and of $\mathcal{SDP}$ (base case, day d1).}",
             r"\label{tab:q3_validation}", r"\begin{tabular}{llp{3.2cm}p{4.2cm}ll}", r"\toprule",
             r"Check & Test & Expected & Measured & Tolerance & Result \\", r"\midrule"]
    lines += [" & ".join(esc(r[k]) for k in ("check", "test", "expected", "measured", "tolerance", "result")) + r" \\"
              for r in rows]
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    tex = folder / "q3_validation.tex"
    tex.write_text("\n".join(lines), encoding="utf-8")
    return csv, tex


def v1_scenarios(com, n: int, seed: int) -> bool:
    scen = sd.draw_scenarios(com, n=n, seed=seed)
    sd.check_scenarios(com, scen)                                         # raises on failure
    spec = sd.make_spec(com)
    th = np.array([[s.theta["profile"][cid] for cid in com.member_ids] for s in scen])
    r = spec["profile"][com.member_ids[0]]
    # the mean of n uniform draws has std (hi - lo) / sqrt(12 n): accept 4 standard deviations on the pooled mean
    pooled_tol = 4 * (r.hi - r.lo) / np.sqrt(12 * th.size)
    ok_mean = abs(th.mean() - r.mean) < pooled_tol
    ok_bounds = r.lo <= th.min() and th.max() <= r.hi
    scaling = sg.draw_scaling_scenarios(sg.set_rule(sd.make_spec(com), "profile", None, n_scenarios=n), seed)
    ok_index = all(abs(scaling["profile"][cid][s.index[("profile", cid)]] - s.theta["profile"][cid]) < 1e-15
                   for s in scen for cid in com.member_ids)
    bigger = sd.draw_scenarios(com, n=2 * n, seed=seed)
    ok_nested = all(bigger[k].theta["profile"] == scen[k].theta["profile"] for k in range(n))
    ok = ok_mean and ok_bounds and ok_index and ok_nested
    return check("V1", f"Scenarios (N={n}, seed {seed})",
                 f"theta in [{r.lo}, {r.hi}], mean {r.mean}; E_min = ref. energy; indices recover draws; nested",
                 f"theta in [{th.min():.3f}, {th.max():.3f}], mean {th.mean():.3f}; all others hold",
                 f"mean +-{pooled_tol:.3f}", ok,
                 f"indices: {ok_index}, nested in N={2 * n}: {ok_nested}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--community", default="base")
    ap.add_argument("--time-limit", type=float, default=300.0)
    ap.add_argument("--n", type=int, default=5, help="in-sample scenarios of V5/V6")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()
    run = lambda k: args.only is None or k in args.only
    TL = args.time_limit
    out = RESULTS_DIR / args.community / "q3"
    out.mkdir(parents=True, exist_ok=True)
    com = load_community(args.community)
    results = {}

    if run("V1"):
        results["V1"] = v1_scenarios(com, args.n, args.seed)

    C = CommunityContractCommunityModel(com).build()
    rC = C.solve()
    lam = internal_price_of_C(C)
    exp_s = sd.expected_scenario(com)
    common = dict(budget_balance=False, reg_reference=lam, time_limit=TL)
    t = time.time()
    dp = DynamicPricingCommunityModel(com, method="SOS1", reg_weight=1e-3, reg_reference=lam, time_limit=TL).build()
    r_dp = dp.solve()
    x_dp = r_dp.hourly["posted_price"].to_numpy()
    print(f"     C cost {rC.objective:.6f} DKK | DP {r_dp.status} {r_dp.objective:.6f} DKK ({time.time() - t:.1f} s)")

    if run("V2"):
        t = time.time()
        r1 = StochasticDynamicPricingModel(com, [exp_s], **common).build().solve()
        d_obj, d_x = abs(r1.objective - r_dp.objective), float(np.abs(r1.prices - x_dp).max())
        results["V2"] = check("V2", "SDP on the expected scenario = DP", f"objective {r_dp.objective:.4f} DKK, same x",
                              f"{r1.objective:.4f} DKK; |d obj| {d_obj:.0e}, max |d x| {d_x:.0e}",
                              f"{TOL:.0e} / 1e-4", r1.status == "OPTIMAL" and d_obj < TOL and d_x < 1e-4,
                              f"{time.time() - t:.1f} s, size {r1.meta['size']}")

    if run("V3"):
        copies = [dataclasses.replace(exp_s, id=k, probability=1 / 3) for k in range(3)]
        t = time.time()
        r3 = StochasticDynamicPricingModel(com, copies, start_prices=x_dp, **common).build().solve()
        d_obj, d_x = abs(r3.objective - r_dp.objective), float(np.abs(r3.prices - x_dp).max())
        results["V3"] = check("V3", "SDP on 3 copies of it = DP", f"objective {r_dp.objective:.4f} DKK, same x",
                              f"{r3.objective:.4f} DKK; |d obj| {d_obj:.0e}, max |d x| {d_x:.0e}",
                              f"{TOL:.0e} / 1e-4", r3.status == "OPTIMAL" and d_obj < TOL and d_x < 1e-4,
                              f"{time.time() - t:.1f} s")

    if run("V4"):
        ev, _ = evaluate_prices(com, x_dp, [exp_s])
        dp_cost = r_dp.meta["welfare_cost"]
        d = abs(ev.cost[0] - dp_cost)
        results["V4"] = check("V4", "Evaluator at DP's prices = DP", f"cost {dp_cost:.4f} DKK",
                              f"{ev.cost[0]:.4f} DKK; |d| {d:.0e}; surplus {ev.surplus[0]:+.2f} DKK", "1e-4", d < 1e-4)

    if run("V5") or run("V6"):
        ins = sd.draw_scenarios(com, n=args.n, seed=args.seed)
        t = time.time()
        r5 = StochasticDynamicPricingModel(com, ins, start_prices=x_dp, **common).build().solve()
        v = r5.meta["validation"]
        if run("V5"):
            ev, _ = evaluate_prices(com, r5.prices, ins)
            diff = ev.cost.to_numpy() - r5.scenarios.cost.to_numpy()
            results["V5"] = check(
                "V5", f"SDP, N={args.n}: residuals and evaluator replay",
                "s*mu = 0, D_is = cost_is; evaluator <= in-model cost",
                f"max s*mu {v['max_s_mu']:.0e}, max |D - cost| {v['max_strong_duality_gap']:.0e}; "
                f"evaluator - in-model in [{diff.min():.3f}, {diff.max():.3f}] DKK",
                "1e-6 / 1% of cost",
                v["max_s_mu"] < 1e-6 and v["max_strong_duality_gap"] < 1e-6
                and diff.max() < 1e-6 and -diff.min() < 0.01 * abs(r5.objective),
                f"{r5.status}, gap {r5.meta['mip_gap']:.1e}, {time.time() - t:.0f} s, size {r5.meta['size']}")
            fig = plot_q3_validation(ins, com.members, com.contract.capacity_limit_kW, r5.scenarios.cost.to_numpy(),
                                     ev.cost.to_numpy(), r_dp.meta["welfare_cost"], out)
            print(f"     figure: {fig}")
        if run("V6"):
            # same start for all three, so that runs stopped at the time limit are comparable
            kw = dict(start_prices=r5.prices, **common)
            rw0 = StochasticDynamicPricingModel(com, ins, objective="cvar", cvar_weight=0.0, **kw).build().solve()
            ra1 = StochasticDynamicPricingModel(com, ins, objective="cvar", cvar_weight=1.0, cvar_alpha=1.0,
                                                **kw).build().solve()
            d0, d1 = abs(rw0.objective - r5.objective), abs(ra1.objective - r5.objective)
            results["V6"] = check("V6", "CVaR limit cases = expectation", f"objective {r5.objective:.4f} DKK",
                                  f"w=0: |d| {d0:.0e}; alpha=1, w=1: |d| {d1:.0e}", "1e-4", d0 < 1e-4 and d1 < 1e-4,
                                  f"{rw0.status}, {ra1.status}")

    csv, tex = write_table(ROWS, out)
    print(f"\n{sum(results.values())}/{len(results)} checks passed: "
          + " ".join(f"{k}={'ok' if ok else 'FAIL'}" for k, ok in results.items()))
    print(f"table: {csv}\n       {tex}")


if __name__ == "__main__":
    main()
