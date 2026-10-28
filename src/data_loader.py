"""Load the input data of Assignment 2: a community of Assignment 1 consumers behind one meter, plus the DSO contract.

Usage
-----
>>> from src.data_loader import load_community
>>> com = load_community("base")                 # params_community_base.json, day d1
>>> com.members[0].energy_price                  # one Assignment 1 InputData per member (same attribute names)
>>> com.contract.capacity_limit_kW               # the hourly cap at the connection point (24 values)
>>> com.contract.individual_capacity_limit_kW["C_E1"]   # the member's own cap in the benchmark model I
>>> com.summary()
>>> load_community("base", day="d4")             # another day of the week (prices, PV, reference profiles and that day's cap)
>>> load_community("my_mix")                     # your own data/params_community_my_mix.json (same format)

Data layout (see the README): ``data/`` holds the two SHARED files of Assignment 1 -- ``appliance_params.json`` (the
catalogue) and ``bus_params.json`` (connection point: prices and tariffs) -- the base community ``params_community_base.json``
(the Assignment 1 case-file format: ``consumers[]`` with ``consumer_id``, ``connection_bus``, ``list_appliances``; only the base
community is shipped -- variations of the member mix are built in code or as your own file of the same format),
``weekly_profiles.json`` (seven days of prices, PV ratios and reference load factors), ``dso_contract_params.json`` (the
contract) and the uncertainty file of Question 3 (read by ``src/scenario_generation.py``).

The :class:`InputData` class is the one of the Assignment 1 starter repository, unchanged, so that the models you wrote
there run on every member as they are. The only difference with Assignment 1 is that a member may have no PV system
(``pv_max_kW = 0`` and zero availability) and that the next-day fields are filled from the following day of the week.

Every member has one flexible load of the Assignment 1 kind -- a reference hourly profile, a maximum hourly deviation of
1 kWh from it (``load_lower_kWh`` / ``load_upper_kWh`` are the resulting hourly bounds, within 0-3 kWh/h), a minimum
daily energy equal to the daily reference energy (the consumer keeps its daily energy and only moves it) -- with a LINEAR
disutility ``linear_disutility`` x |l_t - l_ref_t| whose slope is one of four values: ``FL_L1`` 0.25, ``FL_L2`` 0.5, ``FL_L3`` 1.0,
``FL_L4`` 2.0 DKK/kWh (``InputData.slope_class`` is 1-4, from the most to the least flexible). A member may own a PV system
(``PV_01_S``, 3 kW) and/or a battery (``BAT_01_S``, 2 kWh): ``has_pv``, ``has_battery``; ``archetype`` combines slope class and
assets, e.g. ``"L1+PV+BAT"``. The base community has one member per (slope, asset class): 16 members ``C<k>``, ``C<k>_PV``,
``C<k>_B``, ``C<k>_PVB`` for k = 1..4. Every member's problem is a linear program.
"""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"

SHARED_FILES = ("appliance_params.json", "bus_params.json", "weekly_profiles.json", "dso_contract_params.json")
DAYS = ("d1", "d2", "d3", "d4", "d5", "d6", "d7")


# ======================================================================================================================
# One member = one Assignment 1 consumer (class copied verbatim from the Assignment 1 starter repository)
# ======================================================================================================================
@dataclass
class InputData:
    """All parameters of one consumer for one case, with units in the attribute names."""

    question: str
    consumer_id: str
    hours: np.ndarray                     # 0 .. 23

    # Grid / market conditions (from bus_params.json)
    energy_price: np.ndarray              # DKK/kWh, one value per hour
    import_tariff: float                  # DKK/kWh charged on every kWh imported
    export_tariff: float                  # DKK/kWh charged on every kWh exported
    max_import_kW: float | None           # None = no grid limit
    max_export_kW: float | None           # None = no grid limit

    # PV system (from the "DERs" catalogue)
    pv_max_kW: float
    pv_profile: np.ndarray                # ratio of pv_max_kW available each hour, in [0, 1]
    pv_available: np.ndarray              # kWh/h = pv_max_kW * pv_profile
    pv_marginal_cost: float               # DKK per kWh of PV actually produced (0 if not given)

    # Flexible load (from the "loads" catalogue)
    load_max_kWh: float                   # maximum hourly consumption, kWh/h (connection capacity)
    load_min_kWh: float                   # minimum hourly consumption, kWh/h
    consumption_utility: float | None     # DKK per kWh consumed (Assignment 1 linear-utility load; None here)
    min_daily_energy_kWh: float | None    # minimum energy over the day = the daily reference energy (sum of reference_load)
    reference_load: np.ndarray | None     # preferred hourly consumption, kWh/h
                                          # = load_max_kWh * reference_load_capacity_factor
    linear_disutility: float | None       # DKK per kWh of absolute deviation from reference_load
    quadratic_disutility: float | None    # DKK per kWh^2 of deviation from reference_load
    max_hourly_deviation_kWh: float | None = None   # the load may deviate from reference_load by at most this much in any hour
    load_upper_kWh: np.ndarray | None = None        # hourly upper bound = min(load_max_kWh, reference_load + max_hourly_deviation_kWh)
    load_lower_kWh: np.ndarray | None = None        # hourly lower bound = max(load_min_kWh, reference_load - max_hourly_deviation_kWh)

    # Battery (members with BAT_01; from the "storages" catalogue; all None when absent)
    battery_capacity_kWh: float | None = None
    battery_max_charge_kW: float | None = None
    battery_max_discharge_kW: float | None = None
    battery_charging_efficiency: float | None = None      # fraction of charged energy reaching the battery
    battery_discharging_efficiency: float | None = None   # fraction of discharged energy reaching the consumer
    battery_initial_soc_kWh: float | None = None          # state of charge at the start of the day
    battery_final_soc_kWh: float | None = None            # None ON PURPOSE: the end-of-horizon treatment is your choice

    # Next-day values (the following day of the week; for the end-of-horizon choice)
    energy_price_next_day: np.ndarray | None = None       # DKK/kWh, 24 values for the following day
    pv_available_next_day: np.ndarray | None = None       # kWh/h available the following day
    reference_load_next_day: np.ndarray | None = None     # preferred consumption the following day, kWh/h

    # Everything that was read, unprocessed
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def n_hours(self) -> int:
        return len(self.hours)

    @property
    def slope_class(self) -> int:
        """1-4: the rank of the member's linear disutility slope among the four catalogue values (0.25, 0.5, 1.0, 2.0 DKK/kWh), 1 = most flexible."""
        slopes = [0.25, 0.5, 1.0, 2.0]
        if self.linear_disutility is None:
            return 0
        return 1 + min(range(4), key=lambda j: abs(slopes[j] - self.linear_disutility))

    @property
    def elasticity(self) -> str:
        """``"L<k>"`` with k the slope class (kept for backward compatibility; see ``slope_class``)."""
        return f"L{self.slope_class}"

    @property
    def has_pv(self) -> bool:
        return self.pv_max_kW > 0

    @property
    def has_battery(self) -> bool:
        return self.battery_capacity_kWh is not None

    @property
    def archetype(self) -> str:
        """Slope class and assets, e.g. ``"L1+PV+BAT"``, ``"L3+PV"``, ``"L4"`` (no PV, no battery)."""
        return self.elasticity + ("+PV" if self.has_pv else "") + ("+BAT" if self.has_battery else "")

    @property
    def max_import_capacity_kW(self) -> float:
        """Maximal hourly import: max load plus the battery's charging power (the "capacity" cap-split rule of model I)."""
        return self.load_max_kWh + (self.battery_max_charge_kW or 0.0)

    def summary(self) -> str:
        """Compact textual overview of the member."""
        lines = [
            f"Member               : {self.consumer_id} ({self.archetype})",
            f"Energy price         : mean {self.energy_price.mean():.2f}, min {self.energy_price.min():.2f}, "
            f"max {self.energy_price.max():.2f} DKK/kWh",
            f"Import/export tariff : {self.import_tariff:.2f} / {self.export_tariff:.2f} DKK/kWh (full tariffs of Assignment 1)",
            f"PV                   : " + (f"{self.pv_max_kW:.1f} kW peak, {self.pv_available.sum():.1f} kWh available/day, "
                                          f"marginal cost {self.pv_marginal_cost:.2f} DKK/kWh" if self.pv_max_kW > 0 else "no"),
            f"Load bounds          : {self.load_min_kWh:.1f} - {self.load_max_kWh:.1f} kWh/h"
            + (f", within {self.max_hourly_deviation_kWh:.1f} kWh of the reference in every hour (load_lower_kWh / load_upper_kWh)" if self.max_hourly_deviation_kWh is not None else ""),
            f"Consumption utility  : {self.consumption_utility}",
            f"Min daily energy     : {self.min_daily_energy_kWh}",
            f"Reference profile    : {'yes (%.1f kWh/day)' % self.reference_load.sum() if self.reference_load is not None else 'no'}",
            f"Disutility lin/quad  : {self.linear_disutility} / {self.quadratic_disutility}",
            f"Battery              : "
            + (f"{self.battery_capacity_kWh:.1f} kWh, {self.battery_max_charge_kW:.1f}/"
               f"{self.battery_max_discharge_kW:.1f} kW, eta {self.battery_charging_efficiency:.2f}/"
               f"{self.battery_discharging_efficiency:.2f}, initial SoC {self.battery_initial_soc_kWh:.1f} kWh"
               if self.battery_capacity_kWh is not None else "no"),
        ]
        return "\n".join(lines)


# ======================================================================================================================
# The contract and the community
# ======================================================================================================================
@dataclass
class ContractData:
    """The capacity limitation contract of ``dso_contract_params.json`` for one day."""

    day: str
    tariff_discount_ratio: float                            # beta^DSO: the community pays (1 - beta) x tariffs on its metered flows
    penalty_DKK_per_kWh: float                              # alpha^DSO: per kWh imported above the cap in an hour
    capacity_limit_kW: np.ndarray                           # cap on the community's import at the connection point, 24 values
    capacity_limit_daily_energy_kWh: float                  # = energy factor x daily import of the reference model R on that day
    capacity_limit_daily_energy_factor: float               # 1.3 in the base case (headroom above the import without contract)
    export_capacity_limit_kW: float | None                  # None: no export cap
    share: dict[str, float]                                 # kappa_i of the benchmark model I (sum = 1)
    share_rule: str                                         # "energy_net" (base community: the file's shares) or "capacity" (fallback for another mix)
    net_import_in_R_kWh: dict[str, float] | None            # the quantity the base shares are proportional to (None for another mix: see README)
    max_import_capacity_kW: dict[str, float]                # the quantity of the alternative "capacity" rule
    individual_capacity_limit_kW: dict[str, np.ndarray]     # cap_i,t = kappa_i x cap_t
    alternative_split_rules: dict[str, Any]                 # other ways of splitting the cap (assignment, section Sensitivity analyses)
    reference_model_R: dict[str, float]                     # import, peak, cost, exceedance of R on that day (for checking)
    raw: dict[str, Any] = field(default_factory=dict, repr=False)


@dataclass
class CommunityData:
    """A community of Assignment 1 consumers behind one meter, with its contract, on one day of the week."""

    community: str                                          # "base" (shipped) or the name of your own community file
    community_id: str
    day: str
    day_name: str
    connection_bus: str
    members: list[InputData]
    contract: ContractData
    hyperparameters: dict[str, Any] | None = None
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def member_ids(self) -> list[str]:
        return [m.consumer_id for m in self.members]

    def member(self, consumer_id: str) -> InputData:
        return next(m for m in self.members if m.consumer_id == consumer_id)

    @property
    def energy_price(self) -> np.ndarray:
        """Day-ahead price p_t at the connection point (the same for every member)."""
        return self.members[0].energy_price

    @property
    def import_tariff(self) -> float:
        return self.members[0].import_tariff

    @property
    def export_tariff(self) -> float:
        return self.members[0].export_tariff

    def summary(self) -> str:
        c = self.contract
        arch = sorted({m.archetype for m in self.members})
        counts = ", ".join(f"{a} x{sum(1 for m in self.members if m.archetype == a)}" for a in arch)
        lines = [
            f"Community            : {self.community} ({self.community_id}), day {self.day} - {self.day_name}",
            f"Members              : {len(self.members)}  ({counts})",
            f"Slope classes        : " + ", ".join(f"L{k} x{sum(1 for m in self.members if m.slope_class == k)}" for k in (1, 2, 3, 4)),
            f"Connected load / PV  : {sum(m.load_max_kWh for m in self.members):.0f} kW / "
            f"{sum(m.pv_max_kW for m in self.members):.0f} kW ({sum(1 for m in self.members if m.pv_max_kW > 0)} systems); "
            f"batteries {sum(1 for m in self.members if m.battery_capacity_kWh)}",
            f"Energy price         : mean {self.energy_price.mean():.2f}, min {self.energy_price.min():.2f}, "
            f"max {self.energy_price.max():.2f} DKK/kWh",
            f"Tariffs imp/exp      : {self.import_tariff:.2f} / {self.export_tariff:.2f} DKK/kWh, discount beta = {c.tariff_discount_ratio}",
            f"Cap                  : {c.capacity_limit_kW.min():.2f} - {c.capacity_limit_kW.max():.2f} kW "
            f"(daily energy {c.capacity_limit_daily_energy_kWh:.1f} kWh = {c.capacity_limit_daily_energy_factor} x import without contract), "
            f"penalty {c.penalty_DKK_per_kWh} DKK/kWh",
            f"Individual caps (I)  : shares ({c.share_rule}) " + ", ".join(f"{k} {v:.3f}" for k, v in list(c.share.items())[:4]) + ", ...",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------- loading
def _read_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _first_with(items: list[dict], key: str, value: str, what: str) -> dict:
    for item in items:
        if item.get(key) == value:
            return item
    raise KeyError(f"No {what} with {key} == {value!r}")


def _as_array(values, name: str, n_hours: int = 24) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.shape != (n_hours,):
        raise ValueError(f"{name} must have {n_hours} hourly values, got shape {arr.shape}")
    return arr


def _opt_float(d: dict, key: str) -> float | None:
    v = d.get(key)
    return None if v is None else float(v)


def _community_name(name: str) -> str:
    """Accept ``"base"``, ``"community_base"``, ``"params_community_base"`` or ``"params_community_base.json"``."""
    return name.removesuffix(".json").removeprefix("params_").removeprefix("community_")


def _day_profiles(app: dict, bus: dict, week: dict, day: str) -> tuple[dict, dict]:
    """Copies of the catalogue and the bus entry with the profiles of one day of weekly_profiles.json written into the
    Assignment 1 fields (``energy_price_DKK_per_kWh``, ``hourly_profile_ratio``, ``reference_load_capacity_factor``)."""
    if day not in week["days"]:
        raise KeyError(f"Unknown day {day!r}; available: {list(week['days'])}")
    d = week["days"][day]
    app2, bus2 = copy.deepcopy(app), copy.deepcopy(bus)
    for e in app2.get("DERs") or []:
        if e.get("DER_type") == "PV":
            e["hourly_profile_ratio"] = list(d["pv_hourly_profile_ratio"])
    for e in app2.get("loads") or []:
        if e.get("reference_load_capacity_factor") is not None:
            e["reference_load_capacity_factor"] = list(d["reference_load_capacity_factor"])
            if e.get("min_total_energy_rule"):   # the daily minimum equals the daily reference energy of the day loaded
                e["min_total_energy_per_day_kWh"] = float(e["max_load_KWh_per_hour"] * sum(d["reference_load_capacity_factor"]))
    bus2["energy_price_DKK_per_kWh"] = list(d["energy_price_DKK_per_kWh"])
    return app2, bus2


def _member(consumer: dict, catalogue: dict, bus: dict, next_day: dict | None, case: str) -> InputData:
    cid = consumer["consumer_id"]
    appliance_ids = list(consumer["list_appliances"])
    price = _as_array(bus["energy_price_DKK_per_kWh"], "energy_price_DKK_per_kWh")
    if (price < 0).any():
        raise ValueError("Energy prices are assumed non-negative")

    # --- PV (optional in Assignment 2: two members have none)
    ders = catalogue.get("DERs") or []
    pv = next((d for d in ders if d["DER_id"] in appliance_ids and d["DER_type"] == "PV"), None)
    if pv is not None:
        pv_max = float(pv["max_power_kW"])
        pv_profile = _as_array(pv["hourly_profile_ratio"], "PV hourly_profile_ratio")
        if (pv_profile < 0).any() or (pv_profile > 1).any():
            raise ValueError("PV hourly_profile_ratio must lie in [0, 1]")
        pv_cost = float(pv.get("marginal_cost_DKK_per_kWh") or 0.0)
    else:
        pv_max, pv_profile, pv_cost = 0.0, np.zeros(len(price)), 0.0

    # --- flexible load (exactly one per member)
    loads = catalogue.get("loads") or []
    load = next((l for l in loads if l["load_id"] in appliance_ids), None)
    if load is None:
        raise KeyError(f"None of {appliance_ids} is in the 'loads' catalogue")
    load_max = float(load["max_load_KWh_per_hour"])
    ref_factor = load.get("reference_load_capacity_factor")
    reference_load = None if ref_factor is None else load_max * _as_array(ref_factor, "reference_load_capacity_factor")
    if reference_load is not None and ((reference_load < 0).any() or (reference_load > load_max).any()):
        raise ValueError("reference_load_capacity_factor must lie in [0, 1]")

    # --- battery (members with BAT_01)
    storages = catalogue.get("storages") or []
    bat = next((s for s in storages if s.get("storage_id") in appliance_ids), None)
    bat_kwargs: dict[str, Any] = {}
    if bat is not None:
        cap = float(bat["storage_capacity_kWh"])
        bat_kwargs = dict(
            battery_capacity_kWh=cap,
            battery_max_charge_kW=float(bat["max_charging_power_kW"]),
            battery_max_discharge_kW=float(bat["max_discharging_power_kW"]),
            battery_charging_efficiency=float(bat["charging_efficiency"]),
            battery_discharging_efficiency=float(bat["discharging_efficiency"]),
            battery_initial_soc_kWh=cap * float(bat["initial_state_of_charge_ratio"]),
        )

    # --- next day (the following day of the week), for the end-of-horizon choice
    nd_kwargs: dict[str, Any] = {}
    if next_day is not None:
        nd_kwargs = dict(
            energy_price_next_day=_as_array(next_day["energy_price_DKK_per_kWh"], "next-day price"),
            pv_available_next_day=pv_max * _as_array(next_day["pv_hourly_profile_ratio"], "next-day PV ratio"),
            reference_load_next_day=None if ref_factor is None
                else load_max * _as_array(next_day["reference_load_capacity_factor"], "next-day reference factor"),
        )

    dmax = _opt_float(load, "max_hourly_deviation_kWh")
    return InputData(
        question=case,
        consumer_id=cid,
        hours=np.arange(len(price)),
        energy_price=price,
        import_tariff=float(bus["import_tariff_DKK_per_kWh"]),
        export_tariff=float(bus["export_tariff_DKK_per_kWh"]),
        max_import_kW=_opt_float(bus, "max_import_kW"),
        max_export_kW=_opt_float(bus, "max_export_kW"),
        pv_max_kW=pv_max,
        pv_profile=pv_profile,
        pv_available=pv_max * pv_profile,
        pv_marginal_cost=pv_cost,
        load_max_kWh=load_max,
        load_min_kWh=float(load.get("min_load_KWh_per_hour") or 0.0),
        consumption_utility=_opt_float(load, "consumption_utility_DKK_per_kWh"),
        min_daily_energy_kWh=_opt_float(load, "min_total_energy_per_day_kWh"),
        reference_load=reference_load,
        linear_disutility=_opt_float(load, "linear_disutility_DKK_per_kWh"),
        quadratic_disutility=_opt_float(load, "quadratic_disutility_DKK_per_kWh2"),
        max_hourly_deviation_kWh=dmax,
        load_upper_kWh=(np.minimum(load_max, reference_load + dmax) if (dmax is not None and reference_load is not None) else np.full(len(price), load_max)),
        load_lower_kWh=(np.maximum(float(load.get("min_load_KWh_per_hour") or 0.0), reference_load - dmax) if (dmax is not None and reference_load is not None)
                        else np.full(len(price), float(load.get("min_load_KWh_per_hour") or 0.0))),
        **bat_kwargs,
        **nd_kwargs,
        raw=dict(consumer=consumer, pv=pv, load=load, storage=bat),
    )


def _contract(dso: dict, members: list[InputData], day: str, community_is_base: bool = True) -> ContractData:
    cap = _as_array(dso["capacity_limits_by_day_kW"][day], f"capacity_limits_by_day_kW[{day}]")
    ind = dso["individual_contracts"]
    ids = [m.consumer_id for m in members]
    maxcap = {m.consumer_id: m.max_import_capacity_kW for m in members}
    # Base split rule: proportional to the member's net import in the reference model R on that day. The file gives the
    # shares of the BASE community for every day. For another member mix R changes: recompute the shares with your own
    # model R (the "capacity" rule below needs no model and is defined for every mix; it is only a placeholder).
    share_file = ind.get("share_by_day", {}).get(day, ind["share"])
    net = ind.get("net_import_in_R_kWh_d1") if day == "d1" else None
    if set(share_file) == set(ids) and community_is_base:
        share = {cid: float(share_file[cid]) for cid in ids}; rule = "energy_net"
    else:   # another member mix: R differs from the file's; recompute the net-import shares with your model R (placeholder: capacity rule)
        tot = sum(maxcap.values()); share = {cid: maxcap[cid] / tot for cid in ids}; net = None; rule = "capacity"
    return ContractData(
        day=day,
        tariff_discount_ratio=float(dso["tariff_discount_ratio"]),
        penalty_DKK_per_kWh=float(dso["penalty_DKK_per_kWh"]),
        capacity_limit_kW=cap,
        capacity_limit_daily_energy_kWh=float(cap.sum()),
        capacity_limit_daily_energy_factor=float(dso.get("capacity_limit_daily_energy_factor", 1.0)),
        export_capacity_limit_kW=_opt_float(dso, "export_capacity_limit_kW"),
        share=share, share_rule=rule, net_import_in_R_kWh=net, max_import_capacity_kW=maxcap,
        individual_capacity_limit_kW={cid: share[cid] * cap for cid in ids},
        alternative_split_rules=dict(ind.get("alternative_split_rules", {})),
        reference_model_R=dict(dso.get("reference_model_R_by_day", {}).get(day, {})),
        raw=dso,
    )


def load_community(community: str = "base", day: str = "d1", data_dir: Path | str = DATA_DIR) -> CommunityData:
    """Read the shared files plus ``params_community_<community>.json`` and return a :class:`CommunityData`.

    Only ``"base"`` is shipped. Build variations of the base case in code, on the returned objects (see the assignment,
    section Sensitivity analyses), or write your own ``params_community_<name>.json`` in the same format and load it by name.

    Parameters
    ----------
    community : str
        ``"base"`` (the base case, shipped) or the name of your own community file.
    day : str
        ``"d1"`` (base case, the Assignment 1 day) ... ``"d7"``; the prices, PV ratios and reference load factors of
        ``weekly_profiles.json`` are written into the Assignment 1 fields, so the members' data look exactly as in Assignment 1.
    """
    data_dir = Path(data_dir)
    name = _community_name(community)
    case_file = data_dir / f"params_community_{name}.json"
    if not case_file.exists():
        raise FileNotFoundError(f"Unknown community {community!r} (no {case_file.name} in {data_dir}). Available: {list_communities(data_dir)}")
    raw = {f.removesuffix(".json"): _read_json(data_dir / f) for f in SHARED_FILES}
    raw["params"] = _read_json(case_file)
    week = raw["weekly_profiles"]
    consumers = raw["params"]["consumers"]
    bus_id = consumers[0]["connection_bus"]
    if any(c["connection_bus"] != bus_id for c in consumers):
        raise ValueError("All members of a community share one connection point")
    bus0 = _first_with(raw["bus_params"], "bus_id", bus_id, "bus")
    app, bus = _day_profiles(raw["appliance_params"], bus0, week, day)
    nxt = DAYS[(DAYS.index(day) + 1) % len(DAYS)] if day in DAYS else None
    next_day = week["days"].get(nxt) if nxt else None
    members = [_member(c, app, bus, next_day, f"community_{name}_{day}") for c in consumers]
    contract = _contract(raw["dso_contract_params"], members, day, community_is_base=(name == "base"))
    return CommunityData(
        community=name, community_id=raw["params"].get("community_id", name), day=day, day_name=week["days"][day].get("name", day),
        connection_bus=bus_id, members=members, contract=contract, hyperparameters=raw["params"].get("hyperparameters"), raw=raw,
    )


def list_communities(data_dir: Path | str = DATA_DIR) -> list[str]:
    """Names of all communities under ``data/`` (one per ``params_community_*.json``)."""
    return sorted(p.stem.removeprefix("params_community_") for p in Path(data_dir).glob("params_community_*.json"))


if __name__ == "__main__":  # quick manual check:  python -m src.data_loader
    for name in list_communities():
        print(load_community(name).summary(), end="\n\n")
