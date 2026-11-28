"""Question 1: comparison of R, I and C, contract sensitivity and settlement of C.

The functions here only compute numbers and tables; the figures are drawn in q1_plotting.py and every experiment is
started from main.py using ``python main.py --q1``.
"""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd

from .model import CommunityModel, ContractMode
from .q1_plotting import Question1Plotter

TOL = 1e-6


# --------------------------------------------------------------------------- Helpers
def with_contract(com, beta=None, cap_scale=None):
    """Copy of the community with changed contract terms: discount beta^DSO and/or a multiplier of the cap. The original is not modified."""
    c = copy.deepcopy(com)
    k = c.contract
    if beta is not None:
        k.tariff_discount_ratio = float(beta)
    if cap_scale is not None:
        k.capacity_limit_kW = cap_scale * np.asarray(k.capacity_limit_kW)
        k.individual_capacity_limit_kW = {cid: cap_scale * np.asarray(v)
                                          for cid, v in k.individual_capacity_limit_kW.items()}
    return c


def solve(com, mode: ContractMode, end_value: float):
    """Build and solve one model. Returns the model (variables and duals) and its results."""
    mdl = CommunityModel(com, contract_mode=mode, battery_end_soc_value_DKK_per_kWh=end_value).build()
    return mdl, mdl.solve()


def _arr(var, T):
    return np.array([var[t].X for t in T])


def asset_label(m) -> str:
    """Asset class of a member: none, PV, BAT or PV+BAT."""
    return {(False, False): "none", (True, False): "PV", (False, True): "BAT", (True, True): "PV+BAT"}[(m.has_pv, m.has_battery)]


# --------------------------------------------------------------------------- Section 1.4: Table 3
def dso_metrics(imp: np.ndarray, imp_ref: np.ndarray, cap: np.ndarray) -> dict:
    """The DSO's view: aggregated import (kW) against the cap profile (kW); imp_ref is the import of R."""
    exc = np.maximum(imp - cap, 0.0)
    return {"peak_import_kW": float(imp.max()),
            "total_daily_import_kWh": float(imp.sum()),
            "peak_reduction_pct": float(100 * (imp_ref.max() - imp.max()) / imp_ref.max()),
            "hours_above_cap": int((exc > TOL).sum()),
            "total_exceedance_kWh": float(exc.sum()),
            "max_exceedance_kW": float(exc.max())}


def dso_revenue(mdl, com, mode: ContractMode) -> float:
    """DSO grid revenue (DKK): tariffs plus penalty. R pays the full tariffs; the energy price is a market flow and is excluded."""
    T = list(mdl.T)
    beta = com.contract.tariff_discount_ratio if mode != ContractMode.NONE else 0.0
    alpha = com.contract.penalty_DKK_per_kWh if mode != ContractMode.NONE else 0.0
    if mode == ContractMode.COMMUNITY:
        ti, te = mdl.members[0].import_tariff, mdl.members[0].export_tariff
        imp, exp = _arr(mdl.var["community_import"], T), _arr(mdl.var["community_export"], T)
        exc = _arr(mdl.var["excess"], T)
        return float(((1 - beta) * ti * imp + (1 - beta) * te * exp + (alpha + beta * ti) * exc).sum())
    total = 0.0
    for i, m in enumerate(mdl.members):
        imp = np.array([mdl.var["import"][i, t].X for t in T])
        exp = np.array([mdl.var["export"][i, t].X for t in T])
        exc = np.array([mdl.var["excess"][i, t].X for t in T]) if "excess" in mdl.var else 0.0
        total += float(((1 - beta) * m.import_tariff * imp + (1 - beta) * m.export_tariff * exp
                        + (alpha + beta * m.import_tariff) * exc).sum())
    return total


def base_comparison(com, end_value: float):
    """Comparison of contract types: R, I and C on the base case. For I the aggregated import is the sum of the members' imports.

    Returns (table with one column per model, dict with G_I,R, G_C,R and G_C,I in DKK).
    """
    cap = np.asarray(com.contract.capacity_limit_kW)
    modes = {"R": ContractMode.NONE, "I": ContractMode.INDIVIDUAL, "C": ContractMode.COMMUNITY}
    runs = {tag: solve(com, mode, end_value) for tag, mode in modes.items()}
    J = {tag: res.objective for tag, (_, res) in runs.items()}
    imp_R = runs["R"][1].hourly["import"].to_numpy()
    table = pd.DataFrame({
        tag: {"total_cost_DKK": J[tag],
              **dso_metrics(res.hourly["import"].to_numpy(), imp_R, cap),
              "dso_revenue_DKK": dso_revenue(mdl, com, modes[tag]),
              "gain_vs_R_DKK": J["R"] - J[tag]}
        for tag, (mdl, res) in runs.items()})
    gains = {"G_I,R": J["R"] - J["I"], "G_C,R": J["R"] - J["C"], "G_C,I": J["I"] - J["C"]}
    return table, gains


# --------------------------------------------------------------------------- Sweep of the contract parameters beta^DSO and phi^DSO
def contract_sweep(com, end_value: float, betas, phis) -> pd.DataFrame:
    """Gains and the DSO's view of I and C over the discount beta^DSO and the cap factor phi (alpha^DSO fixed).

    phi defines the cap as phi x (daily import of R) / 24; the base case has phi = 1.3. R does not depend on the contract terms,
    but its exceedance is measured against the cap of each phi. Exceedance is on the aggregated import (kWh/day), revenue is
    tariffs plus penalties (DKK/day), nu_max_C is the largest shadow price of the cap in C (DKK/kWh).
    """
    base_phi = com.contract.capacity_limit_daily_energy_factor
    mR, rR = solve(com, ContractMode.NONE, end_value)
    J_R, imp_R, rev_R = rR.objective, rR.hourly["import"].to_numpy(), dso_revenue(mR, com, ContractMode.NONE)
    rows = []
    for b in betas:
        for phi in phis:
            c = with_contract(com, beta=b, cap_scale=phi / base_phi)
            cap = np.asarray(c.contract.capacity_limit_kW)
            alpha_p = c.contract.penalty_DKK_per_kWh + c.contract.tariff_discount_ratio * c.members[0].import_tariff
            mI, rI = solve(c, ContractMode.INDIVIDUAL, end_value)
            mC, rC = solve(c, ContractMode.COMMUNITY, end_value)
            exc = lambda r: float(np.maximum(r.hourly["import"].to_numpy() - cap, 0.0).sum())
            rows.append({"beta": b, "phi": phi, "gain_I_vs_R": J_R - rI.objective, "gain_C_vs_R": J_R - rC.objective,
                         "pooling_value_C_vs_I": rI.objective - rC.objective,
                         "exc_R": float(np.maximum(imp_R - cap, 0.0).sum()), "exc_I": exc(rI), "exc_C": exc(rC),
                         "rev_R": rev_R, "rev_I": dso_revenue(mI, c, ContractMode.INDIVIDUAL),
                         "rev_C": dso_revenue(mC, c, ContractMode.COMMUNITY),
                         "nu_max_C": float(max(mC.con["excess"][t].Pi for t in mC.T)),
                         # I: the cap constraint's minimization dual is nonpositive; negate it for the cap rent.
                         "nu_max_I": float(max(-c_.Pi for c_ in mI.con["excess"].values())),
                         "nI_at_alpha": int(sum(1 for c_ in mI.con["excess"].values() if -c_.Pi > alpha_p - 1e-6)),
                         })
    return pd.DataFrame(rows)

# --------------------------------------------------------------------------- Settlement of C
def own_costs(mdl, end_value: float) -> dict[str, float]:
    """Costs a member carries independently of the grid payment: PV, disutility, minus the terminal battery value."""
    T = list(mdl.T)
    out = {}
    for i, m in enumerate(mdl.members):
        pv = m.pv_marginal_cost * sum(mdl.var["pv"][i][t].X for t in T) if m.has_pv else 0.0
        dev = m.linear_disutility * sum(mdl.var["deviation"][i, t].X for t in T)
        term = end_value * mdl.var["soc"][i][T[-1]].X if m.has_battery else 0.0
        out[m.consumer_id] = float(pv + dev - term)
    return out


def grid_payment_RI(mdl, com, mode: ContractMode) -> dict[str, float]:
    """Each member's payment to the grid in R (full tariffs) or I (discounted tariffs, own cap and penalty), DKK."""
    T = list(mdl.T)
    p = np.asarray(com.energy_price)
    beta = com.contract.tariff_discount_ratio if mode == ContractMode.INDIVIDUAL else 0.0
    alpha = com.contract.penalty_DKK_per_kWh if mode == ContractMode.INDIVIDUAL else 0.0
    out = {}
    for i, m in enumerate(mdl.members):
        imp = np.array([mdl.var["import"][i, t].X for t in T])
        exp = np.array([mdl.var["export"][i, t].X for t in T])
        exc = np.array([mdl.var["excess"][i, t].X for t in T]) if mode == ContractMode.INDIVIDUAL else 0.0
        out[m.consumer_id] = float(((p + (1 - beta) * m.import_tariff) * imp
                                    - (p - (1 - beta) * m.export_tariff) * exp
                                    + (alpha + beta * m.import_tariff) * exc).sum())
    return out


def community_settlement(mdl, com, end_value: float):
    """Settle model C at the dual price lambda_t of the community balance (the manager keeps the surplus).

    S_i = sum_t (lambda_t q_it + c^PV P^PV_it + c^L d_it) - pi E^bat_i.
    Returns: S as a Series over the members, hourly table with lambda_t, nu_t and the manager's budget B_t.
    """
    T = list(mdl.T)
    p = np.asarray(com.energy_price)
    beta, alpha = com.contract.tariff_discount_ratio, com.contract.penalty_DKK_per_kWh
    ti, te = mdl.members[0].import_tariff, mdl.members[0].export_tariff
    lam = np.array([-mdl.con["community_balance"][t].Pi for t in T])     # internal price lambda_t
    nu = np.array([mdl.con["excess"][t].Pi for t in T])                  # shadow price nu_t of the cap
    q = np.array([[mdl.var["net_consumption"][i, t].X for t in T] for i in mdl.I])
    imp, exp = _arr(mdl.var["community_import"], T), _arr(mdl.var["community_export"], T)
    exc = _arr(mdl.var["excess"], T)
    paid_to_dso = (p + (1 - beta) * ti) * imp - (p - (1 - beta) * te) * exp + (alpha + beta * ti) * exc   # M_t
    budget = lam * q.sum(axis=0) - paid_to_dso                                                          # B_t, equation (24)
    own = own_costs(mdl, end_value)
    ids = [m.consumer_id for m in mdl.members]
    S = pd.Series({cid: float((lam * q[i]).sum() + own[cid]) for i, cid in enumerate(ids)})
    hourly = pd.DataFrame({"lambda": lam, "nu": nu, "budget_B_t": budget}, index=pd.Index(T, name="hour"))
    return S, hourly


def settlement_study(com, end_value: float):
    """Individual rationality of the dual-price settlement of C against R and I, and the manager's budget.

    Returns: members table, hourly table, dict of the numbers.
    """
    mR, rR = solve(com, ContractMode.NONE, end_value)
    mI, _ = solve(com, ContractMode.INDIVIDUAL, end_value)
    mC, rC = solve(com, ContractMode.COMMUNITY, end_value)
    ownR, ownI = own_costs(mR, end_value), own_costs(mI, end_value)
    costR = pd.Series({k: v + ownR[k] for k, v in grid_payment_RI(mR, com, ContractMode.NONE).items()})
    costI = pd.Series({k: v + ownI[k] for k, v in grid_payment_RI(mI, com, ContractMode.INDIVIDUAL).items()})
    S, hourly = community_settlement(mC, com, end_value)

    members = pd.DataFrame({"slope": pd.Series({m.consumer_id: f"L{m.slope_class}" for m in mC.members}),
                            "assets": pd.Series({m.consumer_id: asset_label(m) for m in mC.members}),
                            "cost_R": costR, "cost_I": costI, "S": S})
    members["gain_vs_R"] = members["cost_R"] - members["S"]      # positive = better off under C
    members["gain_vs_I"] = members["cost_I"] - members["S"]

    B = float(hourly["budget_B_t"].sum())
    cap = np.asarray(com.contract.capacity_limit_kW)
    numbers = {
        "J_C": rC.objective, "G_C,R": rR.objective - rC.objective,
        "B (manager surplus)": B,
        "B = sum_t nu_t cap_t": float((hourly["nu"] * cap).sum()),                 # appendix result B_t = nu_t P_cap_t
        "sum_i S_i": float(S.sum()), "J_C + B": rC.objective + B,                    # equation (26)
        "sum_i (J_R,i - S_i)": float(members["gain_vs_R"].sum()),                    # = G_C,R - B
        "members worse off than R": int((members["gain_vs_R"] < -TOL).sum()),
        "members worse off than I": int((members["gain_vs_I"] < -TOL).sum()),
        "sum of costs R / I": (float(costR.sum()), float(costI.sum()))}
    return members, hourly, numbers


class Question1Experiment:
        """Run and report Question 1 experiments for one community."""

        def __init__(self, results_dir):
            self.results_dir = Path(results_dir)

        def run(self, community, args) -> None:
            output_dir = self.results_dir / args.community
            output_dir.mkdir(parents=True, exist_ok=True)
            plotter = Question1Plotter(output_dir)

            if args.q1 or args.q1_base:
                self._run_base(community, args.battery_end_value, output_dir)
            if args.q1 or args.q1_sweep:
                self._run_sweep(community, args.battery_end_value, output_dir, plotter, args.betas, args.phis)
            if args.q1 or args.q1_settlement:
                self._run_settlement(community, args.battery_end_value, output_dir, plotter)

        @staticmethod
        def _run_base(community, end_value: float, output_dir) -> None:
            table, gains = base_comparison(community, end_value)
            table.round(2).to_csv(output_dir / "q1_table3_base_case.csv")
            print("\nTable 3 - base case:\n", table.round(2).to_string())
            print("Gains (DKK):", {key: round(value, 2) for key, value in gains.items()})

        @staticmethod
        def _run_sweep(community, end_value: float, output_dir, plotter, betas, phis) -> None:
            grid = contract_sweep(community, end_value, betas, phis)
            grid.to_csv(output_dir / "q1_sweep_beta_cap.csv", index=False)
            print(
                "\nSmallest value of pooling G_C,I over the grid:",
                round(grid["pooling_value_C_vs_I"].min(), 2),
                "DKK",
            )
            print(f"  {plotter.gain_map(grid)}")
            print("DSO view:", Question1Plotter(output_dir / "sweep").dso_view(grid))

        @staticmethod
        def _run_settlement(community, end_value: float, output_dir, plotter) -> None:
            members, hourly, numbers = settlement_study(community, end_value)
            members.round(4).to_csv(output_dir / "q1_settlement_members.csv")
            hourly.round(4).to_csv(output_dir / "q1_settlement_hourly.csv")
            print(
                "\nSettlement of C:",
                {
                    key: (round(value, 2) if not isinstance(value, tuple) else tuple(round(item, 2) for item in value))
                    for key, value in numbers.items()
                },
            )
            for key in ("assets", "slope"):
                print(f"\nGain vs R by {key} (DKK/day):\n", members.groupby(key)["gain_vs_R"].sum().round(2).to_string())
            print(f"  {plotter.member_gains(members)}")