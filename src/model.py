"""Linear Assignment 1 and Assignment 2 community optimisation models."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import gurobipy as gp
import pandas as pd
from gurobipy import GRB

from .data_loader import CommunityData, InputData


class ContractMode(Enum):
    """Contract applied to the grid exchange."""

    NONE = "none"
    INDIVIDUAL = "individual"
    COMMUNITY = "community"
    DYNAMIC = "dynamic"


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


class SingleConsumerModel:
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

    def build(self) -> "SingleConsumerModel":
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
    """Factory-compatible public entry point for the community model classes."""

    def __new__(
        cls,
        data: CommunityData,
        contract_mode: ContractMode = ContractMode.NONE,
        name: str = "community",
        verbose: bool = False,
        battery_end_soc_value_DKK_per_kWh: float = 0.0,
    ):
        model_class = {
            ContractMode.NONE: NoContractCommunityModel,
            ContractMode.INDIVIDUAL: IndividualContractCommunityModel,
            ContractMode.COMMUNITY: CommunityContractCommunityModel,
            ContractMode.DYNAMIC: DynamicPricingCommunityModel, 
        }[contract_mode]
        return model_class(
            data,
            name=name,
            verbose=verbose,
            battery_end_soc_value_DKK_per_kWh=battery_end_soc_value_DKK_per_kWh,
        )


class _CommunityModelBase:
    """Shared indexed community model behaviour."""

    mode: ContractMode

    def __init__(
        self,
        data: CommunityData,
        name: str = "community",
        verbose: bool = False,
        battery_end_soc_value_DKK_per_kWh: float = 0.0,
    ):
        self.data = data
        self.members = data.members
        self.I = range(len(self.members))
        self.T = range(data.members[0].n_hours)
        self.m = gp.Model(name)
        self.m.Params.OutputFlag = 1 if verbose else 0
        self.terminal_value = battery_end_soc_value_DKK_per_kWh
        self.var: dict[str, Any] = {}
        self.con: dict[str, Any] = {}

    def build(self) -> "_CommunityModelBase":
        self._add_variables()
        self._add_physical_constraints()
        self._add_contract_constraints()
        self.m.setObjective(self._objective(), self._objective_sense())
        self.m.update()
        return self

    def _add_variables(self) -> None:
        m = self.m
        self.var["load"] = m.addVars(self.I, self.T, lb=0, name="load")
        self.var["deviation"] = m.addVars(self.I, self.T, lb=0, name="deviation")
        for i, member in enumerate(self.members):
            if member.has_pv:
                self.var.setdefault("pv", {})[i] = m.addVars(self.T, lb=0, name=f"pv_{i}")
            if member.has_battery:
                for name in ("charge", "discharge", "soc"):
                    self.var.setdefault(name, {})[i] = m.addVars(self.T, lb=0, name=f"{name}_{i}")
        self._add_contract_variables()

    def _add_contract_variables(self) -> None:
        self.var["import"] = self.m.addVars(self.I, self.T, lb=0, name="import")
        self.var["export"] = self.m.addVars(self.I, self.T, lb=0, name="export")

    def _objective(self):
        raise NotImplementedError

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
        self._add_member_constraints()

    def _add_member_constraints(self) -> None:
        m = self.m
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

    def _add_contract_constraints(self) -> None:
        pass

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

    def _battery_terminal_value(self):
        return gp.quicksum(
            -self.terminal_value * self.var["soc"][i][self.members[i].n_hours - 1]
            for i in self.I
            if self.members[i].has_battery
        )

    def _objective_sense(self) -> int:
        return GRB.MINIMIZE

    def _extract_results(self, status: str) -> Results:
        hours = list(self.T)
        hourly = pd.DataFrame(index=pd.Index(hours, name="hour"))
        hourly["import"] = [sum(self.var["import"][i, t].X for i in self.I) for t in hours]
        hourly["export"] = [sum(self.var["export"][i, t].X for i in self.I) for t in hours]
        hourly["load"] = [sum(self.var["load"][i, t].X for i in self.I) for t in hours]
        hourly["pv"] = [
            sum(
                self._member_var("pv", i, t).X
                if hasattr(self._member_var("pv", i, t), "X")
                else 0
                for i in self.I
            )
            for t in hours
        ]
        return Results(
            question=f"{self.mode.value}_community",
            status=status,
            objective=self.m.ObjVal,
            hourly=hourly,
            meta={
                "members": len(self.members),
                "contract_mode": self.mode.value,
                "member_operating_costs": self._member_operating_costs(),
            },
        )

    def _member_operating_costs(self) -> dict[str, float]:
        costs: dict[str, float] = {}
        beta = self.data.contract.tariff_discount_ratio if self.mode != ContractMode.NONE else 0.0
        alpha = self.data.contract.penalty_DKK_per_kWh
        for i, member in enumerate(self.members):
            cost = 0.0
            for t in self.T:
                pv = self._member_var("pv", i, t)
                pv_value = pv.X if hasattr(pv, "X") else 0.0
                deviation = self.var["deviation"][i, t].X
                if self.mode == ContractMode.COMMUNITY:
                    net = self.var["net_consumption"][i, t].X
                    member_import = max(net, 0.0)
                    member_export = max(-net, 0.0)
                    excess = 0.0
                else:
                    member_import = self.var["import"][i, t].X
                    member_export = self.var["export"][i, t].X
                    excess_var = self.var.get("excess", {}).get((i, t), 0.0)
                    excess = excess_var.X if hasattr(excess_var, "X") else 0.0
                cost += (
                    member.pv_marginal_cost * pv_value
                    + (self.data.energy_price[t] + (1.0 - beta) * member.import_tariff) * member_import
                    - (self.data.energy_price[t] - (1.0 - beta) * member.export_tariff) * member_export
                    + (alpha + beta * member.import_tariff) * excess
                    + member.linear_disutility * deviation
                )
            costs[member.consumer_id] = float(cost)
        if self.mode == ContractMode.COMMUNITY:
            total_excess = sum(
                max(self.var["community_import"][t].X - self.data.contract.capacity_limit_kW[t], 0.0)
                for t in self.T
            )
            positive_net = {
                member.consumer_id: sum(
                    max(self.var["net_consumption"][i, t].X, 0.0) for t in self.T
                )
                for i, member in enumerate(self.members)
            }
            total_positive_net = sum(positive_net.values())
            if total_positive_net > 0:
                for consumer_id in costs:
                    costs[consumer_id] += (
                        alpha + beta * self.members[0].import_tariff
                    ) * total_excess * positive_net[consumer_id] / total_positive_net
        return costs

    def solve(self) -> Results:
        self.m.optimize()
        status = _status_name(self.m.Status)
        if self.m.Status != GRB.OPTIMAL:
            raise RuntimeError(f"Optimisation ended with status {status}.")
        return self._extract_results(status)


class NoContractCommunityModel(_CommunityModelBase):
    """Community model where every member pays the full grid tariffs."""

    mode = ContractMode.NONE

    def _objective(self):
        terms = []
        for i, member in enumerate(self.members):
            if member.linear_disutility is None:
                raise ValueError(f"{member.question} does not define linear_disutility.")
            for t in self.T:
                terms.append(
                    member.pv_marginal_cost * self._member_var("pv", i, t)
                    + (self.data.energy_price[t] + member.import_tariff)
                    * self.var["import"][i, t]
                    - (self.data.energy_price[t] - member.export_tariff)
                    * self.var["export"][i, t]
                    + member.linear_disutility * self.var["deviation"][i, t]
                )
        return gp.quicksum(terms) + self._battery_terminal_value()


class IndividualContractCommunityModel(_CommunityModelBase):
    """Community model with one discounted, individually capped contract per member."""

    mode = ContractMode.INDIVIDUAL

    def _add_contract_variables(self) -> None:
        super()._add_contract_variables()
        self.var["excess"] = self.m.addVars(self.I, self.T, lb=0, name="excess")

    def _add_contract_constraints(self) -> None:
        super()._add_contract_constraints()
        caps = self.data.contract.individual_capacity_limit_kW
        self.con["excess"] = self.m.addConstrs(
            (
                self.var["import"][i, t]
                - caps[self.members[i].consumer_id][t] 
                - self.var["excess"][i, t]
                <= 0
                for i in self.I
                for t in self.T
            ),
            name="excess",
        )
        self.con["excess_nonnegative"] = self.m.addConstrs(
            (-self.var["excess"][i, t] <= 0 for i in self.I for t in self.T),
            name="excess_nonnegative",
        )

    def _objective(self):
        c = self.data.contract
        beta = c.tariff_discount_ratio
        alpha = c.penalty_DKK_per_kWh
        terms = []
        for i, member in enumerate(self.members):
            if member.linear_disutility is None:
                raise ValueError(f"{member.question} does not define linear_disutility.")
            for t in self.T:
                terms.append(
                    member.pv_marginal_cost * self._member_var("pv", i, t)
                    + (self.data.energy_price[t] + (1.0 - beta) * member.import_tariff)
                    * self.var["import"][i, t]
                    - (self.data.energy_price[t] - (1.0 - beta) * member.export_tariff)
                    * self.var["export"][i, t]
                    + (alpha + beta * member.import_tariff)
                    * self.var["excess"][i, t]
                    + member.linear_disutility * self.var["deviation"][i, t]
                )
        return gp.quicksum(terms) + self._battery_terminal_value()


class CommunityContractCommunityModel(_CommunityModelBase):
    """Centralized-dispatch community contract at the shared meter."""

    mode = ContractMode.COMMUNITY

    def _add_physical_constraints(self) -> None:
        self.con["net_consumption_balance"] = self.m.addConstrs(
            (
                self.var["net_consumption"][i, t]
                == self.var["load"][i, t]
                + self._member_var("charge", i, t)
                - self._member_var("discharge", i, t)
                - self._member_var("pv", i, t)
                for i in self.I for t in self.T
            ),
            name="net_consumption_balance",
        )
        self._add_member_constraints()

    def _add_contract_variables(self) -> None:
        self.var["net_consumption"] = self.m.addVars(
            self.I, self.T, lb=-GRB.INFINITY, name="net_consumption"
        )
        self.var["community_import"] = self.m.addVars(self.T, lb=0, name="community_import")
        self.var["community_export"] = self.m.addVars(self.T, lb=0, name="community_export")
        self.var["excess"] = self.m.addVars(self.T, lb=0, name="excess")

    def _add_contract_constraints(self) -> None:
        self.con["community_balance"] = self.m.addConstrs(
            (
                gp.quicksum(self.var["net_consumption"][i, t] for i in self.I)
                == self.var["community_import"][t] - self.var["community_export"][t]
                for t in self.T
            ),
            name="community_balance",
        )
        self.con["excess"] = self.m.addConstrs(
            (
                self.var["excess"][t]
                >= self.var["community_import"][t] - self.data.contract.capacity_limit_kW[t]
                for t in self.T
            ),
            name="excess",
        )

    def _extract_results(self, status: str) -> Results:
        hours = list(self.T)
        hourly = pd.DataFrame(index=pd.Index(hours, name="hour"))
        hourly["import"] = [self.var["community_import"][t].X for t in hours]
        hourly["export"] = [self.var["community_export"][t].X for t in hours]
        hourly["load"] = [sum(self.var["load"][i, t].X for i in self.I) for t in hours]
        hourly["pv"] = [
            sum(
                self._member_var("pv", i, t).X
                if hasattr(self._member_var("pv", i, t), "X")
                else 0
                for i in self.I
            )
            for t in hours
        ]
        return Results(
            question=f"{self.mode.value}_community",
            status=status,
            objective=self.m.ObjVal,
            hourly=hourly,
            meta={
                "members": len(self.members),
                "contract_mode": self.mode.value,
                "member_operating_costs": self._member_operating_costs(),
            },
        )

    def _objective(self):
        c = self.data.contract
        beta = c.tariff_discount_ratio
        alpha = c.penalty_DKK_per_kWh
        member_terms = []
        for i, member in enumerate(self.members):
            if member.linear_disutility is None:
                raise ValueError(f"{member.question} does not define linear_disutility.")
            for t in self.T:
                member_terms.append(
                    member.pv_marginal_cost * self._member_var("pv", i, t)
                    + member.linear_disutility * self.var["deviation"][i, t]
                )
        meter_terms = gp.quicksum(
            (self.data.energy_price[t] + (1.0 - beta) * self.members[0].import_tariff)
            * self.var["community_import"][t]
            - (self.data.energy_price[t] - (1.0 - beta) * self.members[0].export_tariff)
            * self.var["community_export"][t]
            + (alpha + beta * self.members[0].import_tariff)
            * self.var["excess"][t]
            for t in self.T
        )
        return gp.quicksum(member_terms) + meter_terms + self._battery_terminal_value()


TOL = 1e-6

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
            lower = d.load_lower_kWh if d.load_lower_kWh is not None else [d.load_min_kWh] * d.n_hours
            upper = d.load_upper_kWh if d.load_upper_kWh is not None else [d.load_max_kWh] * d.n_hours
            L, dev = self.var["load"], self.var["deviation"]

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
                if d.has_battery:
                    phi = D["phi"][i]
                    eta_c, eta_d = d.battery_charging_efficiency, d.battery_discharging_efficiency
                    m.addConstr(x[t] - eta_c * phi[t] - D["ch_lo"][i][t] + D["ch_up"][i][t] == 0, name=f"st_ch_{i}_{t}")
                    m.addConstr(-x[t] + phi[t] / eta_d - D["dis_lo"][i][t] + D["dis_up"][i][t] == 0, name=f"st_dis_{i}_{t}")
                    nxt = (-self.terminal_value + (D["final"][i] if i in D.get("final", {}) else 0)) if t == last else -phi[t + 1]
                    m.addConstr(phi[t] + nxt - D["E_lo"][i][t] + D["E_up"][i][t] == 0, name=f"st_E_{i}_{t}")

            # ---- 5c: complementarity (primal feasibility is already enforced by model C's constraints)
             # ---- per-set big-M bounds (used only when tight_M=True)
            X, cL, cPV, ref = self.X, d.linear_disutility, d.pv_marginal_cost, d.reference_load
            if d.has_battery:
                eta_c, eta_d = d.battery_charging_efficiency, d.battery_discharging_efficiency
                Pc, Pd, S = d.battery_max_charge_kW, d.battery_max_discharge_kW, d.battery_capacity_kWh
                Phi = max(X / eta_c, self.terminal_value)          # largest value of one stored kWh

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
                if d.has_battery:
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
            expr -= gp.quicksum(D["ch_up"][i][t] * d.battery_max_charge_kW
                                + D["dis_up"][i][t] * d.battery_max_discharge_kW
                                + D["E_up"][i][t] * d.battery_capacity_kWh for t in T)
            expr -= D["phi"][i][0] * d.battery_initial_soc_kWh                  # initial state of charge
            if i in D.get("final", {}):
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
        ids = [d.consumer_id for d in self.members]
        zeta = self.m.addVars(self.I, lb=0.0, name="ir_slack")
        self.var["ir_slack"] = zeta
        self.con["individual_rationality"] = {
            i: self.m.addConstr(self.D_expr[i] <= self.ir_benchmark[ids[i]] + zeta[i], name=f"ir_{ids[i]}")
            for i in self.I}
        self.m.update()
        self.m.setObjective(self.m.getObjective() + self.ir_penalty * zeta.sum())

def internal_price_of_C(c_model: CommunityContractCommunityModel) -> list[float]:
    """lambda_t of model C as a positive price (= minus Gurobi's Pi of the community balance)."""
    return [-c_model.con["community_balance"][t].Pi for t in c_model.T]