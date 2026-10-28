"""Linear Assignment 1 and Assignment 2 community optimisation models."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import gurobipy as gp
import pandas as pd
from gurobipy import GRB

from .data_loader import CommunityData, InputData


class ContractMode(Enum):
    """Contract applied to the grid exchange."""

    NONE = "none"
    INDIVIDUAL = "individual"
    COMMUNITY = "community"


@dataclass
class Results:
    """Primal and dual solution of one model run."""

    question: str
    status: str
    objective: float
    hourly: pd.DataFrame
    duals: dict[str, float] = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    def save(self, folder: Path | str, tag: str = "") -> None:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        stem = f"{self.question}{'_' + tag if tag else ''}"
        self.hourly.to_csv(folder / f"{stem}_hourly.csv", index_label="hour")
        with open(folder / f"{stem}_summary.txt", "w", encoding="utf-8") as f:
            f.write(f"status    : {self.status}\nobjective : {self.objective:.4f} DKK\n")
            for name, value in self.duals.items():
                f.write(f"dual[{name}] : {value:.4f}\n")

    def __str__(self) -> str:
        totals = ", ".join(
            f"{name}={self.hourly[name].sum():.1f}"
            for name in ("import", "export", "load", "pv")
            if name in self.hourly
        )
        return f"status: {self.status} | objective: {self.objective:.2f} DKK\n" \
               f"daily totals (kWh): {totals}"


_STATUS = {
    GRB.OPTIMAL: "OPTIMAL",
    GRB.INFEASIBLE: "INFEASIBLE",
    GRB.UNBOUNDED: "UNBOUNDED",
    GRB.INF_OR_UNBD: "INF_OR_UNBD",
    GRB.TIME_LIMIT: "TIME_LIMIT",
    GRB.SUBOPTIMAL: "SUBOPTIMAL",
    GRB.NUMERIC: "NUMERIC",
    GRB.INTERRUPTED: "INTERRUPTED",
}


def _status_name(code: int) -> str:
    return _STATUS.get(code, f"STATUS_{code}")


class ConsumerModel:
    """Assignment 1 model for one consumer."""

    def __init__(
        self,
        data: InputData,
        name: str = "flexible_consumer",
        verbose: bool = False,
        battery_end_soc_value_DKK_per_kWh: float | None = None,
    ):
        self.data = data
        self.T = range(data.n_hours)
        self.m = gp.Model(name)
        self.m.Params.OutputFlag = 1 if verbose else 0
        self.var: dict[str, gp.tupledict] = {}
        self.con: dict[str, gp.tupledict | gp.Constr] = {}
        self.terminal_value = (
            data.battery_end_soc_value_DKK_per_kWh
            if battery_end_soc_value_DKK_per_kWh is None
            else battery_end_soc_value_DKK_per_kWh
        )

    def build(self) -> "ConsumerModel":
        self._add_variables()
        self._add_constraints()
        self.m.setObjective(self._objective(), self._objective_sense())
        self.m.update()
        return self

    def _add_variables(self) -> None:
        d = self.data
        self.var["import"] = self.m.addVars(self.T, lb=0, name="import")
        self.var["export"] = self.m.addVars(self.T, lb=0, name="export")
        self.var["load"] = self.m.addVars(self.T, lb=0, name="load")
        self.var["pv"] = self.m.addVars(self.T, lb=0, name="pv")
        self.var["deviation"] = self.m.addVars(self.T, lb=0, name="deviation")
        if d.has_battery:
            self.var["charge"] = self.m.addVars(self.T, lb=0, name="charge")
            self.var["discharge"] = self.m.addVars(self.T, lb=0, name="discharge")
            self.var["soc"] = self.m.addVars(self.T, lb=0, name="soc")

    def _objective(self):
        d = self.data
        if d.linear_disutility is None:
            raise ValueError(f"{d.question} does not define linear_disutility.")
        expression = gp.quicksum(
            d.pv_marginal_cost * self.var["pv"][t]
            + (d.import_tariff + d.energy_price[t]) * self.var["import"][t]
            - (d.energy_price[t] - d.export_tariff) * self.var["export"][t]
            + d.linear_disutility * self.var["deviation"][t]
            for t in self.T
        )
        if d.has_battery:
            if self.terminal_value is None:
                raise ValueError(
                    "battery_end_soc_value_DKK_per_kWh must be supplied for a battery model."
                )
            expression -= self.terminal_value * self.var["soc"][self.data.n_hours - 1]
        return expression

    def _objective_sense(self) -> int:
        return GRB.MINIMIZE

    def _power_balance(self, t: int):
        charge = self.var["charge"][t] if self.data.has_battery else 0
        discharge = self.var["discharge"][t] if self.data.has_battery else 0
        return self.var["import"][t] + self.var["pv"][t] - self.var["load"][t] - charge + discharge - self.var["export"][t]

    def _add_constraints(self) -> None:
        d, m = self.data, self.m
        self.con["power_balance"] = m.addConstrs(
            (self._power_balance(t) == 0 for t in self.T), name="power_balance"
        )
        lower = d.load_lower_kWh if d.load_lower_kWh is not None else [d.load_min_kWh] * d.n_hours
        upper = d.load_upper_kWh if d.load_upper_kWh is not None else [d.load_max_kWh] * d.n_hours
        self.con["load_lower"] = m.addConstrs(
            (self.var["load"][t] >= lower[t] for t in self.T), name="load_lower"
        )
        self.con["load_upper"] = m.addConstrs(
            (self.var["load"][t] <= upper[t] for t in self.T), name="load_upper"
        )
        self.con["pv_available"] = m.addConstrs(
            (self.var["pv"][t] <= d.pv_available[t] for t in self.T), name="pv_available"
        )
        if d.reference_load is None:
            raise ValueError(f"{d.question} does not define reference_load.")
        self.con["deviation_pos"] = m.addConstrs(
            (self.var["load"][t] - d.reference_load[t] <= self.var["deviation"][t] for t in self.T),
            name="deviation_pos",
        )
        self.con["deviation_neg"] = m.addConstrs(
            (d.reference_load[t] - self.var["load"][t] <= self.var["deviation"][t] for t in self.T),
            name="deviation_neg",
        )
        if d.min_daily_energy_kWh is not None:
            self.con["minimum_daily_energy"] = m.addConstr(
                gp.quicksum(self.var["load"][t] for t in self.T) >= d.min_daily_energy_kWh,
                name="minimum_daily_energy",
            )
        if d.has_battery:
            self._add_battery_constraints()

    def _add_battery_constraints(self) -> None:
        d, m = self.data, self.m
        self.con["charge_limit"] = m.addConstrs(
            (self.var["charge"][t] <= d.battery_max_charge_kW for t in self.T),
            name="charge_limit",
        )
        self.con["discharge_limit"] = m.addConstrs(
            (self.var["discharge"][t] <= d.battery_max_discharge_kW for t in self.T),
            name="discharge_limit",
        )
        self.con["soc_limit"] = m.addConstrs(
            (self.var["soc"][t] <= d.battery_capacity_kWh for t in self.T),
            name="soc_limit",
        )
        self.con["soc_balance"] = m.addConstrs(
            (
                self.var["soc"][t]
                == (d.battery_initial_soc_kWh if t == 0 else self.var["soc"][t - 1])
                + d.battery_charging_efficiency * self.var["charge"][t]
                - self.var["discharge"][t] / d.battery_discharging_efficiency
                for t in self.T
            ),
            name="soc_balance",
        )
        if d.battery_final_soc_kWh is not None:
            self.con["final_soc"] = m.addConstr(
                self.var["soc"][d.n_hours - 1] == d.battery_final_soc_kWh,
                name="final_soc",
            )

    def solve(self) -> Results:
        self.m.optimize()
        status = _status_name(self.m.Status)
        if self.m.Status != GRB.OPTIMAL:
            raise RuntimeError(f"Optimisation ended with status {status}.")
        return self._extract_results(status)

    def _extract_results(self, status: str) -> Results:
        d, hours = self.data, list(self.T)
        hourly = pd.DataFrame(index=pd.Index(hours, name="hour"))
        hourly["price"] = d.energy_price
        hourly["reference_load"] = d.reference_load
        for name, variable in self.var.items():
            hourly[name] = [variable[t].X for t in hours]
        return Results(
            question=d.question,
            status=status,
            objective=self.m.ObjVal,
            hourly=hourly,
            meta={"daily_energy": float(hourly["load"].sum())},
        )


class CommunityModel:
    """Indexed community model with no contract or individual contracts."""

    def __init__(
        self,
        data: CommunityData,
        contract_mode: ContractMode = ContractMode.NONE,
        name: str = "community",
        verbose: bool = False,
        battery_end_soc_value_DKK_per_kWh: float = 0.0,
    ):
        if contract_mode is ContractMode.COMMUNITY:
            raise NotImplementedError("The community contract is not implemented yet.")
        self.data = data
        self.mode = contract_mode
        self.members = data.members
        self.I = range(len(self.members))
        self.T = range(data.members[0].n_hours)
        self.m = gp.Model(name)
        self.m.Params.OutputFlag = 1 if verbose else 0
        self.terminal_value = battery_end_soc_value_DKK_per_kWh
        self.var: dict[str, gp.tupledict] = {}
        self.con: dict[str, gp.tupledict | gp.Constr] = {}

    def build(self) -> "CommunityModel":
        self._add_variables()
        self._add_physical_constraints()
        self._add_contract_constraints()
        self.m.setObjective(self._objective(), GRB.MINIMIZE)
        self.m.update()
        return self

    def _add_variables(self) -> None:
        m = self.m
        self.var["import"] = m.addVars(self.I, self.T, lb=0, name="import")
        self.var["export"] = m.addVars(self.I, self.T, lb=0, name="export")
        self.var["load"] = m.addVars(self.I, self.T, lb=0, name="load")
        self.var["deviation"] = m.addVars(self.I, self.T, lb=0, name="deviation")
        for i, member in enumerate(self.members):
            if member.has_pv:
                self.var.setdefault("pv", {})[i] = m.addVars(self.T, lb=0, name=f"pv_{i}")
            if member.has_battery:
                for name in ("charge", "discharge", "soc"):
                    self.var.setdefault(name, {})[i] = m.addVars(self.T, lb=0, name=f"{name}_{i}")
        if self.mode is ContractMode.INDIVIDUAL:
            self.var["excess"] = m.addVars(self.I, self.T, lb=0, name="excess")

    def _member_var(self, name: str, i: int, t: int):
        return self.var.get(name, {}).get(i, {}).get(t, 0)

    def _add_physical_constraints(self) -> None:
        m = self.m
        self.con["power_balance"] = m.addConstrs(
            (
                self.var["import"][i, t]
                + self._member_var("pv", i, t)
                - self.var["load"][i, t]
                - self._member_var("charge", i, t)
                + self._member_var("discharge", i, t)
                - self.var["export"][i, t]
                == 0
                for i in self.I for t in self.T
            ),
            name="power_balance",
        )
        for i, d in enumerate(self.members):
            lower = d.load_lower_kWh if d.load_lower_kWh is not None else [d.load_min_kWh] * d.n_hours
            upper = d.load_upper_kWh if d.load_upper_kWh is not None else [d.load_max_kWh] * d.n_hours
            self.con.setdefault("load_lower", {})[i] = m.addConstrs(
                (self.var["load"][i, t] >= lower[t] for t in self.T), name=f"load_lower_{i}"
            )
            self.con.setdefault("load_upper", {})[i] = m.addConstrs(
                (self.var["load"][i, t] <= upper[t] for t in self.T), name=f"load_upper_{i}"
            )
            if d.reference_load is None or d.linear_disutility is None:
                raise ValueError(f"{d.question} must define reference_load and linear_disutility.")
            self.con.setdefault("deviation_pos", {})[i] = m.addConstrs(
                (self.var["load"][i, t] - d.reference_load[t] <= self.var["deviation"][i, t] for t in self.T),
                name=f"deviation_pos_{i}",
            )
            self.con.setdefault("deviation_neg", {})[i] = m.addConstrs(
                (d.reference_load[t] - self.var["load"][i, t] <= self.var["deviation"][i, t] for t in self.T),
                name=f"deviation_neg_{i}",
            )
            if d.min_daily_energy_kWh is not None:
                m.addConstr(
                    gp.quicksum(self.var["load"][i, t] for t in self.T) >= d.min_daily_energy_kWh,
                    name=f"minimum_daily_energy_{i}",
                )
            if d.has_pv:
                m.addConstrs(
                    (self.var["pv"][i][t] <= d.pv_available[t] for t in self.T),
                    name=f"pv_available_{i}",
                )
            if d.has_battery:
                self._add_member_battery_constraints(i, d)

    def _add_member_battery_constraints(self, i: int, d: InputData) -> None:
        m = self.m
        m.addConstrs((self.var["charge"][i][t] <= d.battery_max_charge_kW for t in self.T), name=f"charge_limit_{i}")
        m.addConstrs((self.var["discharge"][i][t] <= d.battery_max_discharge_kW for t in self.T), name=f"discharge_limit_{i}")
        m.addConstrs((self.var["soc"][i][t] <= d.battery_capacity_kWh for t in self.T), name=f"soc_limit_{i}")
        m.addConstrs(
            (
                self.var["soc"][i][t]
                == (d.battery_initial_soc_kWh if t == 0 else self.var["soc"][i][t - 1])
                + d.battery_charging_efficiency * self.var["charge"][i][t]
                - self.var["discharge"][i][t] / d.battery_discharging_efficiency
                for t in self.T
            ),
            name=f"soc_balance_{i}",
        )
        if d.battery_final_soc_kWh is not None:
            m.addConstr(self.var["soc"][i][d.n_hours - 1] == d.battery_final_soc_kWh, name=f"final_soc_{i}")

    def _add_contract_constraints(self) -> None: # Needs to have share of the contract capacity limit for each member in the individual contract case and the non negative constraint for the excedance variable.
        if self.mode is ContractMode.NONE:
            return
        caps = self.data.contract.individual_capacity_limit_kW
        self.con["excess"] = self.m.addConstrs(
            (
                self.var["excess"][i, t] >= self.var["import"][i, t] - caps[self.members[i].consumer_id][t]
                for i in self.I for t in self.T
            ),
            name="excess",
        )

    def _objective(self):
        c, p = self.data.contract, self.data.energy_price
        beta = 1.0 if self.mode is ContractMode.NONE else 1.0 - c.tariff_discount_ratio
        terms = []
        for i, d in enumerate(self.members):
            for t in self.T:
                terms.append(
                    d.pv_marginal_cost * self._member_var("pv", i, t)
                    + (p[t] + d.import_tariff*(1-beta)) * self.var["import"][i, t] 
                    - (p[t] - d.export_tariff*(1-beta)) * self.var["export"][i, t]
                    + d.linear_disutility * self.var["deviation"][i, t]
                )
                if self.mode is ContractMode.INDIVIDUAL:
                    terms.append(c.penalty_DKK_per_kWh * self.var["excess"][i, t])
            if d.has_battery:
                terms.append(-self.terminal_value * self.var["soc"][i][self.members[i].n_hours - 1])
        return gp.quicksum(terms)

    def solve(self) -> Results:
        self.m.optimize()
        status = _status_name(self.m.Status)
        if self.m.Status != GRB.OPTIMAL:
            raise RuntimeError(f"Optimisation ended with status {status}.")
        return self._extract_results(status)

    def _extract_results(self, status: str) -> Results:
        hours = list(self.T)
        hourly = pd.DataFrame(index=pd.Index(hours, name="hour"))
        hourly["import"] = [sum(self.var["import"][i, t].X for i in self.I) for t in hours]
        hourly["export"] = [sum(self.var["export"][i, t].X for i in self.I) for t in hours]
        hourly["load"] = [sum(self.var["load"][i, t].X for i in self.I) for t in hours]
        hourly["pv"] = [sum(self._member_var("pv", i, t).X if hasattr(self._member_var("pv", i, t), "X") else 0 for i in self.I) for t in hours]
        return Results(
            question=f"{self.mode.value}_community",
            status=status,
            objective=self.m.ObjVal,
            hourly=hourly,
            meta={"members": len(self.members), "contract_mode": self.mode.value},
        )
