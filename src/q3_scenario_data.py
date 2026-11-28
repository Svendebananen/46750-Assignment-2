"""Question 3: turn the joint preference scenarios of ``src.q3_scenario_generation`` into member data the models can use.

``q3_scenario_generation`` returns, for every joint scenario, the scaling parameters theta of every member (and the scaled
appliance dicts of the catalogue). The models work on :class:`InputData` objects, so this module applies theta to the
members of a loaded :class:`CommunityData`:

    profile:  rho_t -> rho_t ** theta_profile          (rho_t = reference_load_t / load_max_kWh, in [0, 1])
              reference_load   = load_max_kWh * rho_t ** theta_profile                       (kWh/h)
              min_daily_energy = sum_t reference_load_t                                       (kWh, the daily minimum follows)
              load_lower/upper = max(load_min, ref - D) / min(load_max, ref + D)              (kWh/h, D = max_hourly_deviation_kWh)
    cL:       linear_disutility -> linear_disutility * theta_cL                               (DKK/kWh)

theta is applied to the members of the day that was loaded, not to the catalogue (which holds the d1 profiles only), so
the scenarios are consistent with the prices, PV and cap of any day d1-d7. The contract (cap, shares) is not uncertain.

Usage
-----
>>> from src.data_loader import load_community
>>> from src import q3_scenario_data as sd
>>> com = load_community("base")
>>> in_sample = sd.draw_scenarios(com, n=5, seed=1)                # list of Scenario, probability 1/5 each
>>> in_sample[0].members[0].reference_load                         # the member's InputData as scenario 0 sees it
>>> com_s = sd.scenario_community(com, in_sample[0])               # a CommunityData copy: any model of src.model runs on it
>>> sd.check_scenarios(com, in_sample)                             # validation checks of Table 12 (raises if one fails)
>>> sd.save_scenarios(in_sample, "results/base/scenarios_in_N5_seed1.csv")
"""
from __future__ import annotations

import csv
import dataclasses
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import q3_scenario_generation as sg
from .data_loader import CommunityData, InputData


@dataclass
class Scenario:
    """One joint scenario of the community.

    id          : position in its scenario set (0 .. N-1)
    probability : weight in the expectation (1/N for the equiprobable sets of q3_scenario_generation)
    theta       : {quantity: {consumer_id: theta}}, quantity in ("profile", "cL"); 1.0 = base value
    index       : {(quantity, consumer_id): k}, the independent draw k this scenario took for every pair (traceability)
    members     : the members' InputData under this scenario, in the order of ``com.members``
    """

    id: int
    probability: float
    theta: dict[str, dict[str, float]]
    index: dict[tuple[str, str], int]
    members: list[InputData]

    def member(self, consumer_id: str) -> InputData:
        return next(m for m in self.members if m.consumer_id == consumer_id)


# ----------------------------------------------------------------------------------------------------------------------
# One member, one community
# ----------------------------------------------------------------------------------------------------------------------
def apply_theta(member: InputData, theta_profile: float = 1.0, theta_cL: float = 1.0) -> InputData:
    """A copy of ``member`` under the scaling parameters of one scenario.

    Inputs : member (InputData of the loaded day), theta_profile (exponent of the reference capacity factor, -),
             theta_cL (factor on the linear disutility slope, -).
    Output : new InputData with reference_load (kWh/h), min_daily_energy_kWh (kWh), load_lower_kWh / load_upper_kWh (kWh/h)
             and linear_disutility (DKK/kWh) of the scenario; every other field is shared with ``member`` (not copied,
             never modified). theta = 1 returns the member's own values.
    """
    changes: dict = {}
    if member.reference_load is not None and theta_profile != 1.0:
        rho = np.clip(member.reference_load / member.load_max_kWh, 0.0, 1.0)   # clip only removes float noise above 1
        ref = member.load_max_kWh * rho ** theta_profile
        changes["reference_load"] = ref
        if member.min_daily_energy_kWh is not None:      # the daily minimum is the daily reference energy of the scenario
            changes["min_daily_energy_kWh"] = float(ref.sum())
        if member.max_hourly_deviation_kWh is not None:  # the hourly bounds follow the scenario's reference
            dmax = member.max_hourly_deviation_kWh
            changes["load_upper_kWh"] = np.minimum(member.load_max_kWh, ref + dmax)
            changes["load_lower_kWh"] = np.maximum(member.load_min_kWh, ref - dmax)
    if member.linear_disutility is not None and theta_cL != 1.0:
        changes["linear_disutility"] = member.linear_disutility * theta_cL
    return dataclasses.replace(member, **changes)


def scenario_members(com: CommunityData, theta: dict[str, dict[str, float]]) -> list[InputData]:
    """The members of ``com`` under ``theta`` = {quantity: {consumer_id: value}} (missing entries = 1), in community order."""
    return [
        apply_theta(
            m,
            theta.get("profile", {}).get(m.consumer_id, 1.0),
            theta.get("cL", {}).get(m.consumer_id, 1.0),
        )
        for m in com.members
    ]


def scenario_community(com: CommunityData, scenario: Scenario) -> CommunityData:
    """A copy of ``com`` whose members are those of ``scenario`` (contract, prices and PV unchanged), so that every model
    of ``src.model`` can be solved on one scenario as it is (e.g. the members' response at fixed prices out of sample)."""
    return dataclasses.replace(com, members=list(scenario.members))


# ----------------------------------------------------------------------------------------------------------------------
# Scenario sets
# ----------------------------------------------------------------------------------------------------------------------
def make_spec(com: CommunityData, rules: dict | None = None) -> dict:
    """{quantity: {consumer_id: BetaRule}} from preference_uncertainty_params.json (or ``rules``), for the members of ``com``.
    Tune it with ``sg.set_rule(spec, quantity, consumer_id or None, n_scenarios=..., lo=..., hi=..., mean=..., kappa=...)``."""
    rules = rules or sg.load_json("preference_uncertainty_params.json")
    return sg.rules_to_spec(rules, com.member_ids)


def draw_scenarios(com: CommunityData, n: int, seed: int, quantities=("profile",), spec: dict | None = None) -> list[Scenario]:
    """``n`` equiprobable joint scenarios (aligned merging) in which ``quantities`` are uncertain, drawn with ``seed``.

    Inputs : com (the loaded community and day), n (number of joint scenarios), seed (int, report it), quantities (subset of
             ("profile", "cL"); the others stay at theta = 1), spec (optional tuned hyper-parameters, see make_spec).
    Output : list of Scenario. Each (quantity, member) pair has its own random stream, so the same seed with a larger n
             gives the same first draws per pair (nested sets: draw k of a pair does not depend on n).
    """
    spec = spec or make_spec(com)
    for q in sg.QUANTITIES:
        sg.set_rule(spec, q, None, n_scenarios=(n if q in quantities else 1))
    scaling = sg.draw_scaling_scenarios(spec, seed)
    community = {"consumers": [m.raw["consumer"] for m in com.members]}
    joint = sg.merge_scenarios(com.raw["appliance_params"], community, scaling, mode="aligned")
    return [
        Scenario(id=s["id"], probability=s["probability"], theta=s["theta"], index=s["index"],
                 members=scenario_members(com, s["theta"]))
        for s in joint
    ]


def expected_scenario(com: CommunityData) -> Scenario:
    """The deterministic baseline: theta = 1 for every member (the base parameters, the mean of every distribution of θ).
    Note that the mean of θ is not the mean profile: rho ** theta is convex in theta, so E[rho ** theta] >= rho."""
    return Scenario(id=0, probability=1.0, theta={q: {cid: 1.0 for cid in com.member_ids} for q in sg.QUANTITIES},
                    index={}, members=list(com.members))


# ----------------------------------------------------------------------------------------------------------------------
# Validation and export
# ----------------------------------------------------------------------------------------------------------------------
def check_scenarios(com: CommunityData, scenarios: list[Scenario], spec: dict | None = None, tol: float = 1e-9) -> None:
    """Validation checks of Table 12; raises AssertionError on the first failure.
    - probabilities sum to one;
    - every theta lies inside the [lo, hi] of its rule (spec, default the rules file), when the quantity is uncertain;
    - every member's daily minimum equals its scenario's reference energy, and lower <= reference <= upper in every hour;
    - reference_load stays in [0, load_max];
    - theta = 1 reproduces the loaded member exactly."""
    spec = spec or make_spec(com)
    assert abs(sum(s.probability for s in scenarios) - 1.0) < 1e-9, "probabilities do not sum to one"
    for s in scenarios:
        for q, per_member in s.theta.items():
            for cid, th in per_member.items():
                r = spec[q][cid]
                if th != 1.0:
                    assert r.lo - tol <= th <= r.hi + tol, f"scenario {s.id}: theta[{q}][{cid}] = {th} outside [{r.lo}, {r.hi}]"
        for m in s.members:
            ref = m.reference_load
            assert np.all(ref >= -tol) and np.all(ref <= m.load_max_kWh + tol), f"scenario {s.id}, {m.consumer_id}: reference out of [0, max]"
            if m.min_daily_energy_kWh is not None:
                assert abs(m.min_daily_energy_kWh - ref.sum()) < 1e-6, f"scenario {s.id}, {m.consumer_id}: daily minimum != reference energy"
            assert np.all(m.load_lower_kWh <= ref + tol) and np.all(ref <= m.load_upper_kWh + tol), \
                f"scenario {s.id}, {m.consumer_id}: reference outside its hourly bounds"
    for base, same in zip(com.members, scenario_members(com, {})):
        assert np.array_equal(base.reference_load, same.reference_load), "theta = 1 does not reproduce the base member"


def save_scenarios(scenarios: list[Scenario], path: Path | str, seed: int | None = None) -> Path:
    """Write one row per (scenario, member): seed, scenario id, probability, consumer, theta of each quantity, index of the
    draw taken, reference energy (kWh) and disutility slope (DKK/kWh). Enough to reproduce and trace every scenario."""
    path = Path(path)
    if path.suffix.lower() in {".csv", ".txt"} and not path.name.startswith("q3_"):
        path = path.with_name(f"q3_{path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["seed", "scenario", "probability", "consumer_id", "theta_profile", "theta_cL",
                    "index_profile", "index_cL", "reference_energy_kWh", "min_daily_energy_kWh", "linear_disutility_DKK_per_kWh"])
        for s in scenarios:
            for m in s.members:
                cid = m.consumer_id
                w.writerow([seed, s.id, s.probability, cid,
                            s.theta.get("profile", {}).get(cid, 1.0), s.theta.get("cL", {}).get(cid, 1.0),
                            s.index.get(("profile", cid), 0), s.index.get(("cL", cid), 0),
                            float(m.reference_load.sum()), m.min_daily_energy_kWh, m.linear_disutility])
    return path


if __name__ == "__main__":   # quick check:  python -m src.q3_scenario_data
    from .data_loader import load_community

    com = load_community("base")
    scen = draw_scenarios(com, n=5, seed=1)
    check_scenarios(com, scen)
    print(f"{len(scen)} scenarios, all checks passed")
    for s in scen:
        e = sum(m.reference_load.sum() for m in s.members)
        print(f"  scenario {s.id}: community reference energy {e:6.1f} kWh "
              f"(base {sum(m.reference_load.sum() for m in com.members):.1f}), theta C1 = {s.theta['profile']['C1']:.3f}")
