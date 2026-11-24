"""Question 3: numerical experiments.

    python -m src.q3_experiments --step1           # DP's prices replayed on the out-of-sample scenarios (figure 1)

Seeds (report them): in-sample scenarios seed 1 (N = 5, 10), out-of-sample scenarios seed 1000 (100 scenarios), the
same out-of-sample set for every model (common random numbers).

Step 1 (suggested step of the assignment): how do different draws of the profiles change the members' response at the
prices of the deterministic model DP (expected parameters), and in turn the manager's cost, budget and the members'
individual rationality? Writes to results/<community>/q3/:
    step1_dp_oos_scenarios.csv   one row per out-of-sample scenario (cost components, surplus, exceedance)
    step1_dp_oos_members.csv     one row per (scenario, member): cost, benchmark C^R_is, IR violation
    step1_dp_oos_metrics.csv     distribution metrics (mean, quantiles, CVaR, frequencies)
    q3_uncertainty_impact.png    figure 1
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from . import scenario_data as sd
from .data_loader import load_community
from .model import CommunityContractCommunityModel, DynamicPricingCommunityModel, internal_price_of_C
from .plotting import plot_q3_uncertainty_impact
from .stochastic import benchmark_costs, evaluate_prices, risk_metrics

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
SEED_IN, SEED_OUT, N_OUT = 1, 1000, 100


def deterministic_dp(com, time_limit: float | None = None):
    """DP on the expected parameters, regularized towards C's internal prices (as in Question 2)."""
    C = CommunityContractCommunityModel(com).build()
    C.solve()
    lam = internal_price_of_C(C)
    r = DynamicPricingCommunityModel(com, method="SOS1", reg_weight=1e-3, reg_reference=lam,
                                     time_limit=time_limit).build().solve()
    return r, lam


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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--community", default="base")
    ap.add_argument("--step1", action="store_true", help="DP's prices on the out-of-sample scenarios")
    ap.add_argument("--n-out", type=int, default=N_OUT)
    ap.add_argument("--seed-out", type=int, default=SEED_OUT)
    args = ap.parse_args()
    com = load_community(args.community)
    out = RESULTS_DIR / args.community / "q3"
    out.mkdir(parents=True, exist_ok=True)
    if args.step1:
        step1(com, out, args.n_out, args.seed_out)


if __name__ == "__main__":
    main()
