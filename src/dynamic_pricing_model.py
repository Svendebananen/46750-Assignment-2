"""Dynamic-pricing community optimization model and settlement prices."""
from __future__ import annotations

from typing import Any

import gurobipy as gp
from gurobipy import GRB

from .data_loader import InputData
from .model import CommunityContractCommunityModel, ContractMode, Results

TOL = 1e-6


def _battery_parameters(member: InputData) -> tuple[float, float, float, float, float, float]:
    if (
        member.battery_charging_efficiency is None
        or member.battery_discharging_efficiency is None
        or member.battery_max_charge_kW is None
        or member.battery_max_discharge_kW is None
        or member.battery_capacity_kWh is None
        or member.battery_initial_soc_kWh is None
    ):
        raise ValueError(f"{member.question} has incomplete battery parameters.")
    return (
        member.battery_charging_efficiency,
        member.battery_discharging_efficiency,
        member.battery_max_charge_kW,
        member.battery_max_discharge_kW,
        member.battery_capacity_kWh,
        member.battery_initial_soc_kWh,
    )


class DynamicPricingCommunityModel(CommunityContractCommunityModel):
    mode = ContractMode.DYNAMIC

    def __init__(self, data, name="dynamic_pricing", verbose=False,
                battery_end_soc_value_DKK_per_kWh=0.0,
                method="bigM", M_P=50.0, M_D=100.0, reg_weight=0.0, reg_reference=None,
                mip_gap=1e-6, time_limit=None, budget_balance=False,
                ir_benchmark=None, ir_penalty=100.0, tight_M=False, M_scale=1.0, mip_focus=0):
        super().__init__(data, name=name, verbose=verbose,
                         battery_end_soc_value_DKK_per_kWh=battery_end_soc_value_DKK_per_kWh)
        self.method, self.M_P, self.M_D = method, M_P, M_D
        self.X = float(max(data.energy_price)) + self.members[0].import_tariff + data.contract.penalty_DKK_per_kWh
        self.reg_weight, self.reg_reference = reg_weight, reg_reference
        self.pairs = []          # (name, slack, multiplier), used by validate()
        self.dual = {}           # all multipliers, by name
        self.pairs = []          # (name, slack, multiplier, M_P, M_D), used by validate()
        self.tight_M = tight_M   # True: one M^P and one M^D per complementarity set
        self.M_scale = M_scale   # safety factor on the per-set Ms (1.0 = the derived bounds)
        self.budget_balance = budget_balance
        self.D_expr = {}
        self.ir_benchmark = ir_benchmark
        self.ir_penalty = ir_penalty
        self.m.Params.MIPGap = mip_gap         
        self.m.Params.MIPFocus = mip_focus     # 0 balanced, 1 find good solutions first, 3 improve the bound                      
        if time_limit is not None:                                   
            self.m.Params.TimeLimit = time_limit  

    def build(self):
        super().build()              # the WHOLE of model C
        self._add_price()            # x_t
        self._add_kkt()              # stationarity + complementarity: creates the multipliers
        if self.budget_balance or self.ir_benchmark is not None:
            self.D_expr = {i: self.dual_objective(i) for i in self.I}   # needs the multipliers
        if self.budget_balance:
            self._add_budget_balance()
        if self.ir_benchmark is not None:
            self._add_individual_rationality()
        if self.reg_weight > 0:
            self._add_regularization()
        self.m.update()
        return self

    def _add_price(self):
        self.var["price"] = self.m.addVars(self.T, lb=0.0, ub=self.X, name="x")

    def _compl(self, expr, mu, name, M_P=None, M_D=None):
        """0 <= s = expr  _|_  mu >= 0. M_P, M_D: this set's own bounds, used when tight_M=True."""
        if self.tight_M and M_P is not None and M_D is not None:
            MP, MD = max(self.M_scale * M_P, 1e-6), max(self.M_scale * M_D, 1e-6)
        else:
            MP, MD = self.M_P, self.M_D
        s = self.m.addVar(lb=0.0, name=f"s_{name}")
        self.m.addConstr(s == expr)                      # s = the constraint's slack
        if self.method == "SOS1":
            self.m.addSOS(GRB.SOS_TYPE1, [s, mu])
        else:                                            # big-M with one binary
            u = self.m.addVar(vtype=GRB.BINARY, name=f"u_{name}")
            self.m.addConstr(s  <= MP * u)               # u = 0  ->  s = 0 (constraint binds)
            self.m.addConstr(mu <= MD * (1 - u))         # u = 1  ->  mu = 0 (constraint slack)
        self.pairs.append((name, s, mu, MP, MD))

    def _add_kkt(self) -> None:
        m, x, T, last = self.m, self.var["price"], self.T, max(self.T)
        mv = lambda key: m.addVars(self.I, T, lb=0.0, name=key)          # non-negative multipliers
        D = self.dual
        for key in ("dev_up", "dev_lo", "L_lo", "L_up"):
            D[key] = mv(key)
        D["min"] = m.addVars(self.I, lb=0.0, name="mu_min")

        for i, d in enumerate(self.members):
            if d.linear_disutility is None or d.reference_load is None:
                raise ValueError(f"{d.question} must define reference_load and linear_disutility.")
            lower = d.load_lower_kWh if d.load_lower_kWh is not None else [d.load_min_kWh] * d.n_hours
            upper = d.load_upper_kWh if d.load_upper_kWh is not None else [d.load_max_kWh] * d.n_hours
            L, dev = self.var["load"], self.var["deviation"]
            battery = _battery_parameters(d) if d.has_battery else None

            # ---- 5a: multipliers of this member's PV and battery
            # PV: hours without sun have both PV bounds binding (degenerate) -> no pair, PV is already forced to 0
            sunny = [t for t in T if d.has_pv and d.pv_available[t] > TOL]
            if d.has_pv:
                D.setdefault("PV_lo", {})[i] = m.addVars(sunny, lb=0.0, name=f"PV_lo_{i}")
                D.setdefault("PV_up", {})[i] = m.addVars(sunny, lb=0.0, name=f"PV_up_{i}")
            if d.has_battery:
                for key in ("ch_lo", "ch_up", "dis_lo", "dis_up", "E_lo", "E_up"):
                    D.setdefault(key, {})[i] = m.addVars(T, lb=0.0, name=f"{key}_{i}")
                D.setdefault("phi", {})[i] = m.addVars(T, lb=-GRB.INFINITY, name=f"phi_{i}")
                if d.battery_final_soc_kWh is not None:
                    D.setdefault("final", {})[i] = m.addVar(lb=-GRB.INFINITY, name=f"final_{i}")

            # ---- 5b: stationarity (lambda_{i,t} = x_t)
            for t in T:
                m.addConstr(x[t] + D["dev_up"][i, t] - D["dev_lo"][i, t]
                            - D["L_lo"][i, t] + D["L_up"][i, t] - D["min"][i] == 0, name=f"st_L_{i}_{t}")
                m.addConstr(d.linear_disutility - D["dev_up"][i, t] - D["dev_lo"][i, t] == 0, name=f"st_d_{i}_{t}")
                if t in sunny:
                    m.addConstr(d.pv_marginal_cost - x[t] - D["PV_lo"][i][t] + D["PV_up"][i][t] == 0,
                                name=f"st_PV_{i}_{t}")
                if battery is not None:
                    phi = D["phi"][i]
                    eta_c, eta_d = battery[:2]
                    m.addConstr(x[t] - eta_c * phi[t] - D["ch_lo"][i][t] + D["ch_up"][i][t] == 0, name=f"st_ch_{i}_{t}")
                    m.addConstr(-x[t] + phi[t] / eta_d - D["dis_lo"][i][t] + D["dis_up"][i][t] == 0, name=f"st_dis_{i}_{t}")
                    nxt = (-self.terminal_value + (D["final"][i] if i in D.get("final", {}) else 0)) if t == last else -phi[t + 1]
                    m.addConstr(phi[t] + nxt - D["E_lo"][i][t] + D["E_up"][i][t] == 0, name=f"st_E_{i}_{t}")

            # ---- 5c: complementarity (primal feasibility is already enforced by model C's constraints)
             # ---- per-set big-M bounds (used only when tight_M=True)
            X, cL, cPV, ref = self.X, d.linear_disutility, d.pv_marginal_cost, d.reference_load
            # ---- 5c: complementarity (primal feasibility is already enforced by model C's constraints)
            #      self._compl(slack, multiplier, name, M^P, M^D)
            for t in T:
                n = f"{i}_{t}"
                dev_MP = 2 * max(upper[t] - ref[t], ref[t] - lower[t])     # slack = 2|L - Lref|
                rng = upper[t] - lower[t]                                   # hourly load range
                self._compl(dev[i, t] - L[i, t] + ref[t], D["dev_up"][i, t], "devup_" + n, dev_MP, cL)
                self._compl(dev[i, t] + L[i, t] - ref[t], D["dev_lo"][i, t], "devlo_" + n, dev_MP, cL)
                self._compl(L[i, t] - lower[t], D["L_lo"][i, t], "Llo_" + n, rng, X + cL)
                self._compl(upper[t] - L[i, t], D["L_up"][i, t], "Lup_" + n, rng, X + 2 * cL)
                if t in sunny:
                    pv = self.var["pv"][i][t]
                    self._compl(pv, D["PV_lo"][i][t], "PVlo_" + n, d.pv_available[t], cPV)
                    self._compl(d.pv_available[t] - pv, D["PV_up"][i][t], "PVup_" + n, d.pv_available[t], X - cPV)
                if battery is not None:
                    eta_c, eta_d, Pc, Pd, S, _ = battery
                    Phi = max(X / eta_c, self.terminal_value)
                    ch, dis, soc = self.var["charge"][i][t], self.var["discharge"][i][t], self.var["soc"][i][t]
                    self._compl(ch, D["ch_lo"][i][t], "chlo_" + n, Pc, X)
                    self._compl(Pc - ch, D["ch_up"][i][t], "chup_" + n, Pc, eta_c * Phi)
                    self._compl(dis, D["dis_lo"][i][t], "dislo_" + n, Pd, Phi / eta_d)
                    self._compl(Pd - dis, D["dis_up"][i][t], "disup_" + n, Pd, X)
                    self._compl(soc, D["E_lo"][i][t], "Elo_" + n, S, Phi)
                    self._compl(S - soc, D["E_up"][i][t], "Eup_" + n, S, Phi)
            if d.min_daily_energy_kWh is not None:
                self._compl(gp.quicksum(L[i, t] for t in T) - d.min_daily_energy_kWh, D["min"][i], f"Emin_{i}",
                            sum(upper) - d.min_daily_energy_kWh, X + cL)


    def dual_objective(self, i: int):
        """D_i, member i's dual objective: every multiplier times the constant of its constraint.
        By strong duality D_i equals member i's own cost, but D_i is linear."""
        d, D, T = self.members[i], self.dual, self.T
        if d.reference_load is None:
            raise ValueError(f"{d.question} does not define reference_load.")
        lower = d.load_lower_kWh if d.load_lower_kWh is not None else [d.load_min_kWh] * d.n_hours
        upper = d.load_upper_kWh if d.load_upper_kWh is not None else [d.load_max_kWh] * d.n_hours
        expr = gp.quicksum(
            (D["dev_lo"][i, t] - D["dev_up"][i, t]) * d.reference_load[t]     # deviation constraints
            + D["L_lo"][i, t] * lower[t] - D["L_up"][i, t] * upper[t]          # load bounds
            for t in T)
        if d.min_daily_energy_kWh is not None:
            expr += D["min"][i] * d.min_daily_energy_kWh                        # daily minimum
        if d.has_pv:
            expr -= gp.quicksum(D["PV_up"][i][t] * d.pv_available[t] for t in D["PV_up"][i])
        if d.has_battery:
            _, _, max_charge, max_discharge, capacity, initial_soc = _battery_parameters(d)
            expr -= gp.quicksum(D["ch_up"][i][t] * max_charge
                                + D["dis_up"][i][t] * max_discharge
                                + D["E_up"][i][t] * capacity for t in T)
            expr -= D["phi"][i][0] * initial_soc                  # initial state of charge
            if i in D.get("final", {}) and d.battery_final_soc_kWh is not None:
                expr -= D["final"][i] * d.battery_final_soc_kWh
        return expr

    def _add_budget_balance(self) -> None:
        """sum_i payment_i = meter bill  <=>  sum_i D_i = objective (10)."""
        self.con["budget_balance"] = self.m.addConstr(
            gp.quicksum(self.D_expr.values()) == self._objective(), name="budget_balance")


    def _add_regularization(self) -> None:
        """Pick one price among equally good ones: stay close to a reference (C's duals)."""
        beta = self.data.contract.tariff_discount_ratio
        ref = self.reg_reference if self.reg_reference is not None else [
            self.data.energy_price[t] + (1 - beta) * self.members[0].import_tariff for t in self.T]
        z = self.m.addVars(self.T, lb=0.0, name="price_dev")
        self.m.addConstrs((z[t] >= self.var["price"][t] - ref[t] for t in self.T))
        self.m.addConstrs((z[t] >= ref[t] - self.var["price"][t] for t in self.T))
        self.m.update()                       
        self.m.setObjective(self.m.getObjective() + self.reg_weight * z.sum())
        self.var["price_dev"] = z

    def _member_operating_costs(self) -> dict[str, float]:
        x = [self.var["price"][t].X for t in self.T]
        costs = {}
        for i, d in enumerate(self.members):
            cost = 0.0
            for t in self.T:
                pv = self._member_var("pv", i, t)
                cost += (x[t] * self.var["net_consumption"][i, t].X
                         + d.pv_marginal_cost * (pv.X if hasattr(pv, "X") else 0.0)
                         + d.linear_disutility * self.var["deviation"][i, t].X)
            if d.has_battery:
                cost -= self.terminal_value * self.var["soc"][i][max(self.T)].X
            costs[d.consumer_id] = float(cost)
        return costs

    def _extract_results(self, status: str) -> Results:
        res = super()._extract_results(status)        
        res.hourly["posted_price"] = [self.var["price"][t].X for t in self.T]
        res.meta["validation"] = self.validate()
        res.meta["binaries"], res.meta["sos"] = self.m.NumBinVars, self.m.NumSOS
        res.meta["mip_gap"] = self.m.MIPGap
        reg = self.reg_weight * sum(self.var["price_dev"][t].X for t in self.T) if self.reg_weight > 0 else 0.0
        res.meta["regularization_cost"] = reg
        ir_pen = self.ir_penalty * sum(self.var["ir_slack"][i].X for i in self.I) if self.ir_benchmark is not None else 0.0
        res.meta["welfare_cost"] = res.objective - reg - ir_pen   # compare THIS with C
        res.meta["manager_surplus"] = sum(res.meta["member_operating_costs"].values()) - res.meta["welfare_cost"]
        if self.D_expr:                                                                                   
            costs = res.meta["member_operating_costs"]                                                    
            res.meta["validation"]["max_strong_duality_gap"] = max(                                       
                abs(self.D_expr[i].getValue() - costs[d.consumer_id]) for i, d in enumerate(self.members))
        if self.ir_benchmark is not None:
            costs = res.meta["member_operating_costs"]
            res.meta["individual_rationality"] = {
                d.consumer_id: {"benchmark": self.ir_benchmark[d.consumer_id], "cost_DP": costs[d.consumer_id],
                                "slack": self.var["ir_slack"][i].X}
                for i, d in enumerate(self.members)}
        return res

    def solve(self) -> Results:
        self.m.optimize()
        if self.m.SolCount == 0:
            raise RuntimeError(f"No solution, Gurobi status {self.m.Status}. "
                               "With big-M, try SOS1: INFEASIBLE often means an M that is too small.")
        return self._extract_results("OPTIMAL" if self.m.Status == GRB.OPTIMAL else f"STATUS_{self.m.Status}")

    def validate(self) -> dict[str, Any]:
        """Course checks: complementarity, distance to M, prices at their bounds, indifferent pairs."""
        x = [self.var["price"][t].X for t in self.T]
        return {
            "max_complementarity_residual": max((s.X * mu.X for _, s, mu, _, _ in self.pairs), default=0.0),
            "max_s_over_MP": max(s.X / MP for _, s, _, MP, _ in self.pairs),
            "max_mu_over_MD": max(mu.X / MD for _, _, mu, _, MD in self.pairs),
            # sets whose multiplier sits on its own M^D (deviation sets left out: there mu = c^L = M^D is normal)
            "sets_at_M_D": sorted({n.split("_")[0] for n, _, mu, _, MD in self.pairs
                                   if mu.X > MD * (1 - 1e-6) and not n.startswith("dev")}),
            "price_bound_X": self.X,
            "hours_price_at_bound": [t for t in self.T if x[t] < TOL or x[t] > self.X - TOL],
            "indifferent_pairs": [n for n, s, mu, _, _ in self.pairs if s.X < TOL and mu.X < TOL],
        }

    def _add_individual_rationality(self) -> None:
        """Soft individual rationality:  D_i <= C_i^R + zeta_i,  zeta_i >= 0,  for every member i.
        zeta_i = how much worse off member i is than in R = the compensation it needs to join."""
        benchmark = self.ir_benchmark
        if benchmark is None:
            raise ValueError("An individual-rationality benchmark is required.")
        ids = [d.consumer_id for d in self.members]
        zeta = self.m.addVars(self.I, lb=0.0, name="ir_slack")
        self.var["ir_slack"] = zeta
        self.con["individual_rationality"] = {
            i: self.m.addConstr(self.D_expr[i] <= benchmark[ids[i]] + zeta[i], name=f"ir_{ids[i]}")
            for i in self.I}
        self.m.update()
        self.m.setObjective(self.m.getObjective() + self.ir_penalty * zeta.sum())

def internal_price_of_C(c_model: CommunityContractCommunityModel) -> list[float]:
    """lambda_t of model C as a positive price (= minus Gurobi's Pi of the community balance)."""
    return [-c_model.con["community_balance"][t].Pi for t in c_model.T]