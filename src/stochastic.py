"""Question 3: stochastic pricing SDP^x and the evaluation of a posted price vector on (unseen) scenarios.

One parametrized model, ``StochasticDynamicPricingModel``: one price vector x_t (first stage) and, for every joint
scenario s, the whole of model DP (model C's flows + every member's KKT system) with the scenario's member data
(second stage). The variants differ in two switches:

    objective   = "expectation"   min  sum_s pi_s C_s                                            (SDP^x, risk neutral)
                  "cvar"          min  (1 - w) sum_s pi_s C_s + w (v + 1/a sum_s pi_s u_s)          (SDP^CVaR)
                                       u_s >= C_s - v, u_s >= 0, v free;  w = cvar_weight, a = cvar_alpha (tail share)
    constraints = "expectation"   sum_s pi_s B_s = 0,   sum_s pi_s D_is <= sum_s pi_s C^R_is + zeta_i   (SDP^x)
                  "chance"        b_s = 0  ->  B_s >= 0  and  D_is <= C^R_is + zeta_is,   sum_s pi_s b_s <= epsilon
                                                                                                    (SDP^eps)
                                  epsilon = 0 enforces both in every scenario (no binaries b_s)

    C_s  = M_s + sum_i (D_is - sum_t x_t q_ist)          community cost of scenario s (model C's objective)
    B_s  = sum_i D_is - C_s = sum_it x_t q_ist - M_s     manager surplus of scenario s (payments - meter bill)
    D_is = member i's dual objective in scenario s       (= its cost, by strong duality; linear)

The objective also carries the price regularization rho sum_t |x_t - x_t^ref| and the IR penalty kappa^IR x zeta.
The deterministic baseline DP over the expected parameters is the same class with ``[sd.expected_scenario(com)]``.

``evaluate_prices`` replays a fixed x on any scenario set: every member's LP is solved at x (the members are
independent given x) and, among the members' optimal responses, the one that is best for the community is kept
(optimistic tie-break, the one the MPEC assumes in-sample). Out-of-sample results are therefore comparable with the
in-sample ones.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import gurobipy as gp
import numpy as np
import pandas as pd
from gurobipy import GRB

from . import scenario_data as sd
from .data_loader import CommunityData
from .model import (CommunityContractCommunityModel, DynamicPricingCommunityModel, IndividualContractCommunityModel,
                    NoContractCommunityModel)

OBJECTIVES = ("expectation", "cvar")
CONSTRAINTS = ("expectation", "chance")


# ----------------------------------------------------------------------------------------------------------------------
# One scenario of SDP: model C + KKT systems of the members, inside a shared Gurobi model and with a shared price
# ----------------------------------------------------------------------------------------------------------------------
class _ScenarioBlock(DynamicPricingCommunityModel):
    """The lower levels of one scenario. Reuses DP's KKT and dual objective, but adds them to the shared model ``m``
    and uses the shared price ``price`` instead of creating its own; budget balance, IR and regularization are added by
    the stochastic model on top of all the blocks."""

    def __init__(self, data: CommunityData, m: gp.Model, price: gp.tupledict, s: int, **dp_kwargs):
        super().__init__(data, name=f"scenario_{s}", **dp_kwargs)
        self.m.dispose()                       # DP's own model is not used: everything goes into the shared model
        self.m, self.price, self.s = m, price, s

    def build(self) -> "_ScenarioBlock":
        self._add_variables()
        self._add_physical_constraints()
        self._add_contract_constraints()
        self.var["price"] = self.price
        self._add_kkt()
        self.D_expr = {i: self.dual_objective(i) for i in self.I}
        return self

    def community_cost(self):
        """C_s: model C's objective (meter bill + PV cost + discomfort - battery end value)."""
        return self._objective()

    def meter_bill(self):
        """M_s: what the manager pays at the connection point (energy, discounted tariffs, penalty)."""
        c, t0 = self.data.contract, self.members[0]
        beta, alpha = c.tariff_discount_ratio, c.penalty_DKK_per_kWh
        p, v = self.data.energy_price, self.var
        return gp.quicksum((p[t] + (1 - beta) * t0.import_tariff) * v["community_import"][t]
                           - (p[t] - (1 - beta) * t0.export_tariff) * v["community_export"][t]
                           + (alpha + beta * t0.import_tariff) * v["excess"][t] for t in self.T)

    def payments(self):
        """sum_it x_t q_ist: what the members pay the manager."""
        return gp.quicksum(self.price[t] * self.var["net_consumption"][i, t] for i in self.I for t in self.T)


# ----------------------------------------------------------------------------------------------------------------------
# Results
# ----------------------------------------------------------------------------------------------------------------------
@dataclass
class StochasticResult:
    """One solve of SDP. ``scenarios``: one row per scenario; ``members``: one row per (scenario, member)."""

    name: str
    status: str
    objective: float
    prices: np.ndarray
    scenarios: pd.DataFrame
    members: pd.DataFrame
    hourly: dict[str, pd.DataFrame] = field(default_factory=dict)   # per-scenario import/export/excess/load
    meta: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        s, m = self.scenarios, self.meta
        return (f"{self.name}: {self.status}, objective {self.objective:.3f} DKK, gap {m['mip_gap']:.2e}, "
                f"{m['runtime_s']:.1f} s | E[C] {s['cost'].mean():.3f}, max C {s['cost'].max():.3f}, "
                f"E[surplus] {s['surplus'].mean():+.3f}, min surplus {s['surplus'].min():+.3f}, "
                f"E[exceedance] {s['exceedance_kWh'].mean():.3f} kWh")


def _status(m: gp.Model) -> str:
    return {GRB.OPTIMAL: "OPTIMAL", GRB.TIME_LIMIT: "TIME_LIMIT", GRB.INFEASIBLE: "INFEASIBLE",
            GRB.SUBOPTIMAL: "SUBOPTIMAL", GRB.INTERRUPTED: "INTERRUPTED"}.get(m.Status, f"STATUS_{m.Status}")


# ----------------------------------------------------------------------------------------------------------------------
# The stochastic pricing model
# ----------------------------------------------------------------------------------------------------------------------
class StochasticDynamicPricingModel:
    """SDP: one price vector, one lower level (C's flows + the members' KKT systems) per scenario.

    com            : the loaded community (prices, PV, contract); the members of each scenario come from ``scenarios``
    scenarios      : list of ``scenario_data.Scenario`` (probabilities must sum to one)
    objective      : "expectation" or "cvar" (cvar_alpha = share of worst scenarios, cvar_weight = w in [0, 1])
    constraints    : "expectation" or "chance" (epsilon = allowed probability of violation; 0 = every scenario)
    budget_balance : impose the budget-balance requirement (in the sense of ``constraints``)
    ir_benchmark   : None (no IR) or one {consumer_id: cost} per scenario, the C^R_is (see ``benchmark_costs``)
    ir_penalty     : kappa^IR, DKK per DKK of IR slack
    reg_weight     : rho, weight of sum_t |x_t - reg_reference_t| (reg_reference default: discounted import price)
    start_prices   : optional x_t for a two-phase MIP start (see ``warm_start``), e.g. the deterministic DP's prices
    method         : "SOS1" (default, exact) or "bigM" (tight_M, M_scale, M_P, M_D as in DynamicPricingCommunityModel)
    """

    def __init__(self, com: CommunityData, scenarios: list[sd.Scenario], *, name: str = "SDP",
                 objective: str = "expectation", cvar_alpha: float = 0.2, cvar_weight: float = 0.5,
                 constraints: str = "expectation", epsilon: float = 0.0,
                 budget_balance: bool = True, ir_benchmark: list[dict[str, float]] | None = None,
                 ir_penalty: float = 100.0, reg_weight: float = 1e-3, reg_reference=None,
                 method: str = "SOS1", tight_M: bool = True, M_scale: float = 1.0, M_P: float = 50.0,
                 M_D: float = 100.0, battery_end_soc_value_DKK_per_kWh: float = 0.0,
                 mip_gap: float = 1e-4, time_limit: float | None = None, mip_focus: int = 0,
                 start_prices=None, threads: int = 0, verbose: bool = False):
        if objective not in OBJECTIVES or constraints not in CONSTRAINTS:
            raise ValueError(f"objective in {OBJECTIVES}, constraints in {CONSTRAINTS}")
        if abs(sum(s.probability for s in scenarios) - 1.0) > 1e-9:
            raise ValueError("scenario probabilities must sum to one")
        if ir_benchmark is not None and len(ir_benchmark) != len(scenarios):
            raise ValueError("ir_benchmark needs one {consumer_id: cost} per scenario")
        if not 0.0 < cvar_alpha <= 1.0 or not 0.0 <= cvar_weight <= 1.0:
            raise ValueError("cvar_alpha in (0, 1], cvar_weight in [0, 1]")
        self.com, self.scenarios, self.name = com, scenarios, name
        self.S = range(len(scenarios))
        self.pi = [s.probability for s in scenarios]
        self.T = range(com.members[0].n_hours)
        self.objective_mode, self.cvar_alpha, self.cvar_weight = objective, cvar_alpha, cvar_weight
        self.constraints_mode, self.epsilon = constraints, epsilon
        self.budget_balance, self.ir_benchmark, self.ir_penalty = budget_balance, ir_benchmark, ir_penalty
        self.reg_weight, self.reg_reference = reg_weight, reg_reference
        self.start_prices = start_prices
        self.terminal_value = battery_end_soc_value_DKK_per_kWh
        self.dp_kwargs = dict(method=method, tight_M=tight_M, M_scale=M_scale, M_P=M_P, M_D=M_D,
                              battery_end_soc_value_DKK_per_kWh=battery_end_soc_value_DKK_per_kWh)
        self.m = gp.Model(name)
        self.m.Params.OutputFlag = 1 if verbose else 0
        self.m.Params.MIPGap = mip_gap
        self.m.Params.MIPFocus = mip_focus
        if threads:
            self.m.Params.Threads = threads
        if time_limit is not None:
            self.m.Params.TimeLimit = time_limit
        self.blocks: list[_ScenarioBlock] = []
        self.var: dict[str, Any] = {}
        self.expr: dict[str, Any] = {}

    # ---- build -------------------------------------------------------------------------------------------------------
    def build(self) -> "StochasticDynamicPricingModel":
        t0 = time.perf_counter()
        X = float(max(self.com.energy_price)) + self.com.members[0].import_tariff + self.com.contract.penalty_DKK_per_kWh
        self.X = X
        self.var["price"] = self.m.addVars(self.T, lb=0.0, ub=X, name="x")
        for s, scen in zip(self.S, self.scenarios):
            com_s = sd.scenario_community(self.com, scen)
            self.blocks.append(_ScenarioBlock(com_s, self.m, self.var["price"], s, **self.dp_kwargs).build())
        self.expr["C"] = [b.community_cost() for b in self.blocks]
        self.expr["D"] = [b.D_expr for b in self.blocks]
        self.expr["B"] = [gp.quicksum(b.D_expr.values()) - b.community_cost() for b in self.blocks]
        self._add_requirements()
        self.m.setObjective(self._objective(), GRB.MINIMIZE)
        self.m.update()
        self.build_time = time.perf_counter() - t0
        return self

    def _chance(self) -> bool:
        return self.constraints_mode == "chance" and self.epsilon > 0

    def _add_requirements(self) -> None:
        m, S, pi = self.m, self.S, self.pi
        ids = [d.consumer_id for d in self.com.members]
        I = range(len(ids))
        if self.constraints_mode == "expectation":
            if self.budget_balance:
                self.con_bb = m.addConstr(gp.quicksum(pi[s] * self.expr["B"][s] for s in S) == 0, name="bb_expected")
            if self.ir_benchmark is not None:
                zeta = self.var["zeta"] = m.addVars(I, lb=0.0, name="zeta")
                for i, cid in enumerate(ids):
                    m.addConstr(gp.quicksum(pi[s] * (self.expr["D"][s][i] - self.ir_benchmark[s][cid]) for s in S)
                                <= zeta[i], name=f"ir_expected_{cid}")
            return
        # chance constraints (epsilon = 0: in every scenario, no binaries)
        if self._chance():
            b = self.var["b"] = m.addVars(S, vtype=GRB.BINARY, name="b")
            m.addConstr(gp.quicksum(pi[s] * b[s] for s in S) <= self.epsilon, name="violation_probability")
        if self.ir_benchmark is not None:
            self.var["zeta"] = m.addVars(S, I, lb=0.0, name="zeta")
        for s in S:
            rows = []
            if self.budget_balance:
                rows.append((self.expr["B"][s], GRB.GREATER_EQUAL, f"bb_{s}"))
            if self.ir_benchmark is not None:
                for i, cid in enumerate(ids):
                    rows.append((self.expr["D"][s][i] - self.ir_benchmark[s][cid] - self.var["zeta"][s, i],
                                 GRB.LESS_EQUAL, f"ir_{s}_{cid}"))
            for lhs, sense, nm in rows:
                if self._chance():      # b_s = 0  ->  requirement holds; b_s = 1 -> scenario s may violate it
                    m.addGenConstrIndicator(self.var["b"][s], False, lhs, sense, 0.0, name=nm)
                elif sense == GRB.GREATER_EQUAL:
                    m.addConstr(lhs >= 0, name=nm)
                else:
                    m.addConstr(lhs <= 0, name=nm)

    def _objective(self):
        m, S, pi, C = self.m, self.S, self.pi, self.expr["C"]
        expected = gp.quicksum(pi[s] * C[s] for s in S)
        if self.objective_mode == "cvar":
            v = self.var["var"] = m.addVar(lb=-GRB.INFINITY, name="VaR")
            u = self.var["u"] = m.addVars(S, lb=0.0, name="u")
            m.addConstrs((u[s] >= C[s] - v for s in S), name="cvar_excess")
            self.expr["CVaR"] = v + (1.0 / self.cvar_alpha) * gp.quicksum(pi[s] * u[s] for s in S)
            risk = (1 - self.cvar_weight) * expected + self.cvar_weight * self.expr["CVaR"]
        else:
            risk = expected
        self.expr["risk"] = risk
        obj = risk
        if self.reg_weight > 0:
            beta = self.com.contract.tariff_discount_ratio
            ref = self.reg_reference if self.reg_reference is not None else [
                self.com.energy_price[t] + (1 - beta) * self.com.members[0].import_tariff for t in self.T]
            z = self.var["price_dev"] = m.addVars(self.T, lb=0.0, name="price_dev")
            m.addConstrs((z[t] >= self.var["price"][t] - ref[t] for t in self.T))
            m.addConstrs((z[t] >= ref[t] - self.var["price"][t] for t in self.T))
            self.expr["reg"] = self.reg_weight * z.sum()
            obj = obj + self.expr["reg"]
        if self.ir_benchmark is not None:
            zeta = self.var["zeta"]
            if self.constraints_mode == "expectation":
                self.expr["ir_pen"] = self.ir_penalty * zeta.sum()
            else:
                n = len(self.com.members)
                self.expr["ir_pen"] = self.ir_penalty * gp.quicksum(pi[s] * zeta[s, i] for s in S for i in range(n))
            obj = obj + self.expr["ir_pen"]
        return obj

    # ---- size --------------------------------------------------------------------------------------------------------
    def size(self) -> dict[str, int]:
        m = self.m
        return {"scenarios": len(self.scenarios), "members": len(self.com.members), "variables": m.NumVars,
                "binaries": m.NumBinVars, "constraints": m.NumConstrs, "sos": m.NumSOS,
                "indicators": m.NumGenConstrs, "complementarity_pairs": sum(len(b.pairs) for b in self.blocks)}

    # ---- solve -------------------------------------------------------------------------------------------------------
    def warm_start(self, prices, time_limit: float = 60.0) -> bool:
        """Two-phase start: fix x to ``prices`` and solve what is left (each scenario's KKT-consistent response, plus
        the requirements), then hand the whole solution to Gurobi as a MIP start and release x. Returns False (and
        falls back to a start on x only) if no solution is found at these prices, e.g. when they violate budget
        balance or the chance constraint; pass requirement-feasible prices in that case."""
        m, x = self.m, self.var["price"]
        prices = [min(max(float(v), 0.0), self.X) for v in prices]
        old_limit = m.Params.TimeLimit
        for t in self.T:
            x[t].LB = x[t].UB = prices[t]
        m.Params.TimeLimit = min(time_limit, old_limit)
        m.optimize()
        found = m.SolCount > 0
        self.warm_start_info = {"found": found, "status": _status(m), "objective": m.ObjVal if found else None,
                                "runtime_s": m.Runtime}
        start = [(v, v.X) for v in m.getVars()] if found else [(x[t], prices[t]) for t in self.T]
        for t in self.T:
            x[t].LB, x[t].UB = 0.0, self.X
        m.Params.TimeLimit = old_limit
        m.reset()
        for v, val in start:
            v.Start = val
        m.update()
        return found

    def solve(self) -> StochasticResult:
        if self.start_prices is not None:
            self.warm_start(self.start_prices)
        self.m.optimize()
        if self.m.SolCount == 0:
            raise RuntimeError(f"{self.name}: no solution, Gurobi status {self.m.Status}")
        return self._extract()

    def _extract(self) -> StochasticResult:
        x = np.array([self.var["price"][t].X for t in self.T])
        com, c = self.com, self.com.contract
        cap = np.asarray(c.capacity_limit_kW)
        beta, alpha = c.tariff_discount_ratio, c.penalty_DKK_per_kWh
        ti, te = com.members[0].import_tariff, com.members[0].export_tariff
        p = np.asarray(com.energy_price)
        rows, mrows, hourly = [], [], {}
        max_cs, max_sd, max_eq = 0.0, 0.0, 0.0
        for s, b in zip(self.S, self.blocks):
            imp = np.array([b.var["community_import"][t].X for t in self.T])
            exp = np.array([b.var["community_export"][t].X for t in self.T])
            exc = np.array([b.var["excess"][t].X for t in self.T])
            costs = b._member_operating_costs()            # each member's cost at x, from the primal
            meter = b.meter_bill().getValue()
            pay = b.payments().getValue()
            C = self.expr["C"][s].getValue()
            v = b.validate()
            sd_gap = max(abs(b.D_expr[i].getValue() - costs[d.consumer_id]) for i, d in enumerate(b.members))
            max_cs = max(max_cs, v["max_complementarity_residual"])
            max_sd = max(max_sd, sd_gap)
            max_eq = max(max_eq, float(np.max(np.abs(exc - np.maximum(imp - cap, 0.0)))))
            row = {"scenario": s, "probability": self.pi[s], "cost": C, "meter_bill": meter, "payments": pay,
                   "surplus": pay - meter, "energy": float(p @ (imp - exp)),
                   "tariffs": float((1 - beta) * (ti * imp.sum() + te * exp.sum())),
                   "penalty": float((alpha + beta * ti) * np.maximum(imp - cap, 0.0).sum()),
                   "exceedance_kWh": float(np.maximum(imp - cap, 0.0).sum()),
                   "hours_above_cap": int((imp - cap > 1e-6).sum()), "peak_kW": float(imp.max()),
                   "reference_energy_kWh": float(sum(d.reference_load.sum() for d in b.members)),
                   "max_s_mu": v["max_complementarity_residual"], "max_strong_duality_gap": sd_gap}
            row["discomfort_pv"] = C - meter
            if "b" in self.var:
                row["b"] = int(round(self.var["b"][s].X))
            rows.append(row)
            for i, d in enumerate(b.members):
                bench = self.ir_benchmark[s][d.consumer_id] if self.ir_benchmark is not None else np.nan
                if "zeta" in self.var:
                    z = self.var["zeta"][s, i].X if self.constraints_mode == "chance" else self.var["zeta"][i].X
                else:
                    z = np.nan
                mrows.append({"scenario": s, "member": d.consumer_id, "cost": costs[d.consumer_id],
                              "benchmark": bench, "ir_violation": max(costs[d.consumer_id] - bench, 0.0),
                              "zeta": z})
            hourly[s] = pd.DataFrame({"import": imp, "export": exp, "excess": exc,
                                      "load": [sum(b.var["load"][i, t].X for i in b.I) for t in self.T]})
        scen = pd.DataFrame(rows)
        meta = {"runtime_s": self.m.Runtime, "build_s": self.build_time,
                "warm_start": getattr(self, "warm_start_info", None), "mip_gap": self.m.MIPGap,
                "mip_bound": self.m.ObjBound, "size": self.size(), "objective_mode": self.objective_mode,
                "constraints_mode": self.constraints_mode, "epsilon": self.epsilon, "cvar_alpha": self.cvar_alpha,
                "cvar_weight": self.cvar_weight, "budget_balance": self.budget_balance,
                "ir": self.ir_benchmark is not None, "price_bound_X": self.X,
                "risk_term": self.expr["risk"].getValue(),
                "reg_term": self.expr["reg"].getValue() if "reg" in self.expr else 0.0,
                "ir_penalty_term": self.expr["ir_pen"].getValue() if "ir_pen" in self.expr else 0.0,
                "validation": {"max_s_mu": max_cs, "max_strong_duality_gap": max_sd,
                               "max_excess_definition_error": max_eq,
                               "expected_surplus": float((scen["surplus"] * scen["probability"]).sum())}}
        if "CVaR" in self.expr:
            meta["VaR"], meta["CVaR"] = self.var["var"].X, self.expr["CVaR"].getValue()
        return StochasticResult(name=self.name, status=_status(self.m), objective=self.m.ObjVal, prices=x,
                                scenarios=scen, members=pd.DataFrame(mrows), hourly=hourly, meta=meta)


# ----------------------------------------------------------------------------------------------------------------------
# Benchmarks and out-of-sample evaluation
# ----------------------------------------------------------------------------------------------------------------------
def benchmark_costs(com: CommunityData, scenarios: list[sd.Scenario], rule: str = "R",
                    battery_end_soc_value_DKK_per_kWh: float = 0.0) -> list[dict[str, float]]:
    """C^bench_is: each member's cost without the community contract in every scenario (R, I, or the better of both).
    Model R (and I) is separable over the members, so solving it on the scenario's community gives every C^R_is."""
    out = []
    for scen in scenarios:
        com_s = sd.scenario_community(com, scen)
        kw = dict(battery_end_soc_value_DKK_per_kWh=battery_end_soc_value_DKK_per_kWh)
        cR = NoContractCommunityModel(com_s, **kw).build().solve().meta["member_operating_costs"]
        if rule == "R":
            out.append(cR)
            continue
        cI = IndividualContractCommunityModel(com_s, **kw).build().solve().meta["member_operating_costs"]
        out.append(cI if rule == "I" else {k: min(cR[k], cI[k]) for k in cR})
    return out


def _member_cost(model: CommunityContractCommunityModel, i: int, x) -> gp.LinExpr:
    d = model.members[i]
    e = gp.quicksum(x[t] * model.var["net_consumption"][i, t] + d.linear_disutility * model.var["deviation"][i, t]
                    for t in model.T)
    if d.has_pv:
        e += gp.quicksum(d.pv_marginal_cost * model.var["pv"][i][t] for t in model.T)
    if d.has_battery:
        e -= model.terminal_value * model.var["soc"][i][max(model.T)]
    return e


def respond(com_s: CommunityData, prices, battery_end_soc_value_DKK_per_kWh: float = 0.0,
            tie_break: str = "optimistic", tol: float = 1e-6) -> CommunityContractCommunityModel:
    """The members' response to the posted prices in one scenario, as a solved model C.

    Step 1: every member minimizes its own cost at x (one LP; the members are independent given x, so the sum is
            minimized member by member) -> D_i*.
    Step 2: among the optimal responses (cost_i <= D_i* + tol), keep the one with the lowest community cost
            ("optimistic", the response the MPEC picks in-sample) or the highest ("pessimistic")."""
    model = CommunityContractCommunityModel(com_s, name="response",
                                            battery_end_soc_value_DKK_per_kWh=battery_end_soc_value_DKK_per_kWh).build()
    m = model.m
    costs = [_member_cost(model, i, prices) for i in model.I]
    m.setObjective(gp.quicksum(costs), GRB.MINIMIZE)
    m.optimize()
    if m.Status != GRB.OPTIMAL:
        raise RuntimeError(f"member response: status {m.Status}")
    best = [c.getValue() for c in costs]
    for i in model.I:
        m.addConstr(costs[i] <= best[i] + tol * (1 + abs(best[i])), name=f"optimal_response_{i}")
    m.setObjective(model._objective(), GRB.MINIMIZE if tie_break == "optimistic" else GRB.MAXIMIZE)
    m.optimize()
    if m.Status != GRB.OPTIMAL:
        raise RuntimeError(f"tie-break: status {m.Status}")
    model.member_cost_expr = costs
    return model


def evaluate_prices(com: CommunityData, prices, scenarios: list[sd.Scenario],
                    benchmark: list[dict[str, float]] | None = None, battery_end_soc_value_DKK_per_kWh: float = 0.0,
                    tie_break: str = "optimistic", label: str = "") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Realized outcome of a fixed price vector on every scenario (in or out of sample).

    Returns (scenarios, members): one row per scenario with the community cost and its components, the manager's
    surplus (budget balance: < 0 is a deficit) and the cap exceedance; one row per (scenario, member) with the member's
    cost and its IR violation against ``benchmark`` (> 0 = worse off than without the contract)."""
    c = com.contract
    cap = np.asarray(c.capacity_limit_kW)
    beta, alpha = c.tariff_discount_ratio, c.penalty_DKK_per_kWh
    ti, te = com.members[0].import_tariff, com.members[0].export_tariff
    p, x = np.asarray(com.energy_price), np.asarray(prices, dtype=float)
    rows, mrows = [], []
    for s, scen in enumerate(scenarios):
        model = respond(sd.scenario_community(com, scen), x, battery_end_soc_value_DKK_per_kWh, tie_break)
        imp = np.array([model.var["community_import"][t].X for t in model.T])
        exp = np.array([model.var["community_export"][t].X for t in model.T])
        net = np.array([[model.var["net_consumption"][i, t].X for t in model.T] for i in model.I])
        over = np.maximum(imp - cap, 0.0)
        meter = float((p + (1 - beta) * ti) @ imp - (p - (1 - beta) * te) @ exp + (alpha + beta * ti) * over.sum())
        pay = float((net @ x).sum())
        cost = model.m.ObjVal
        rows.append({"model": label, "scenario": s, "probability": scen.probability, "cost": cost,
                     "meter_bill": meter, "payments": pay, "surplus": pay - meter,
                     "energy": float(p @ (imp - exp)), "tariffs": float((1 - beta) * (ti * imp.sum() + te * exp.sum())),
                     "penalty": float((alpha + beta * ti) * over.sum()), "discomfort_pv": cost - meter,
                     "exceedance_kWh": float(over.sum()), "hours_above_cap": int((over > 1e-6).sum()),
                     "peak_kW": float(imp.max()),
                     "reference_energy_kWh": float(sum(d.reference_load.sum() for d in scen.members))})
        for i, d in enumerate(model.members):
            ci = model.member_cost_expr[i].getValue()
            bench = benchmark[s][d.consumer_id] if benchmark is not None else np.nan
            mrows.append({"model": label, "scenario": s, "member": d.consumer_id, "cost": ci, "benchmark": bench,
                          "ir_violation": max(ci - bench, 0.0) if benchmark is not None else np.nan})
        model.m.dispose()
    return pd.DataFrame(rows), pd.DataFrame(mrows)


def risk_metrics(scen: pd.DataFrame, members: pd.DataFrame | None = None, alpha: float = 0.2,
                 tol: float = 1e-4) -> dict[str, float]:
    """Distribution metrics of a scenario table (equiprobable or weighted): mean, quantiles and CVaR of the cost,
    frequency and size of budget deficits, of IR violations and of cap exceedance."""
    w = scen["probability"].to_numpy() / scen["probability"].sum()
    cost = scen["cost"].to_numpy()
    order = np.argsort(cost)[::-1]                             # worst first
    cw, tail, acc = cost[order], 0.0, 0.0
    for c_, w_ in zip(cw, w[order]):                           # CVaR_alpha: mean of the worst alpha share
        take = min(w_, alpha - acc)
        if take <= 0:
            break
        tail += take * c_
        acc += take
    out = {"E_cost": float(w @ cost), "q50_cost": float(np.quantile(cost, 0.5)),
           "q90_cost": float(np.quantile(cost, 0.9)), "max_cost": float(cost.max()),
           f"CVaR{int(alpha * 100)}_cost": tail / acc,
           "E_surplus": float(w @ scen["surplus"]), "min_surplus": float(scen["surplus"].min()),
           "P_deficit": float(w @ (scen["surplus"] < -tol)),
           "E_deficit": float(w @ np.maximum(-scen["surplus"], 0.0)),
           "E_exceedance_kWh": float(w @ scen["exceedance_kWh"]), "P_exceedance": float(w @ (scen["exceedance_kWh"] > tol)),
           "E_penalty": float(w @ scen["penalty"])}
    if members is not None and members["ir_violation"].notna().any():
        per_s = members.groupby("scenario")["ir_violation"]
        viol = per_s.max().reindex(scen["scenario"]).to_numpy()
        out["P_ir_violation"] = float(w @ (viol > tol))                         # any member worse off in s
        out["E_ir_violation"] = float(w @ per_s.sum().reindex(scen["scenario"]).to_numpy())   # DKK, summed over members
        out["member_violation_rate"] = float((members["ir_violation"] > tol).mean())
    return out
