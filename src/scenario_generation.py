"""
46750 - Assignment 2: scenario generation for Question 3.

This module implements, exactly, the random rules written in ``preference_uncertainty_params.json`` (Question 3),
so that every group draws scenarios the same way and every experiment is reproducible from a seed. It only depends on ``numpy`` and the standard library.
You may use it as is, extend it, or re-implement it - in all cases state the seeds you used.

Question 3 - uncertain preferences (two-stage, no time dimension)
    rules_to_spec(rules, consumer_ids)                 -> {quantity: {consumer_id: BetaRule}} hyper-parameters from the rules file
    set_rule(spec, quantity, consumer_id, **kw)        -> tune n_scenarios / lo / hi / mean / kappa of one pair (or all consumers)
    draw_scaling_scenarios(spec, seed)                 -> independent scenarios of every scaling parameter, per quantity and consumer
    merge_scenarios(app, community, scaling, mode)     -> joint scenarios of the community (members' perturbed appliance dicts), with the
                                                          index of the scenario taken from every (quantity, consumer) and the theta values
    draw_preference_scenarios(app, community, rules, n, seed, quantities=("profile",))  -> the one-call version of the three above
    expected_preference_scenario(app, community)       -> the catalogue parameters unchanged (the deterministic baseline)


Data helpers
    day_data(app, bus, week, day)                      -> (appliance dict, bus dict) with the profiles of one day of weekly_profiles.json

Run ``python -m src.scenario_generation`` (from the repository root) for a short demonstration.
"""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent / "data"   # the data folder of the repository


def load_json(name: str, folder: Path = DATA_DIR) -> dict:
    with open(folder / name, encoding="utf-8") as f:
        return json.load(f)


# ----------------------------------------------------------------------------------------------------------------------
# Data helpers
# ----------------------------------------------------------------------------------------------------------------------

def day_data(app: dict, bus: list | dict, week: dict, day: str = "d1") -> tuple[dict, dict]:
    """Copies of the appliance catalogue and of the bus entry with the day-ahead prices, PV availability ratios and reference load
    capacity factors of one day of weekly_profiles.json written into the Assignment 1 fields (energy_price_DKK_per_kWh,
    hourly_profile_ratio, reference_load_capacity_factor), so that an Assignment 1 loader can be used unchanged."""
    d = week["days"][day]
    app2 = copy.deepcopy(app)
    for e in app2.get("DERs") or []:
        if e.get("DER_type") == "PV":
            e["hourly_profile_ratio"] = list(d["pv_hourly_profile_ratio"])
    for e in app2.get("loads") or []:
        if e.get("reference_load_capacity_factor") is not None:
            e["reference_load_capacity_factor"] = list(d["reference_load_capacity_factor"])
            if e.get("min_total_energy_rule"):   # the daily minimum equals the daily reference energy of the day
                e["min_total_energy_per_day_kWh"] = float(e["max_load_KWh_per_hour"] * sum(d["reference_load_capacity_factor"]))
    b = copy.deepcopy(bus[0] if isinstance(bus, list) else bus)
    b["energy_price_DKK_per_kWh"] = list(d["energy_price_DKK_per_kWh"])
    return app2, b



# ----------------------------------------------------------------------------------------------------------------------
# Question 3 - preference uncertainty
#
# Four uncertain quantities per consumer, each driven by ONE scaling parameter theta drawn from a scaled Beta distribution:
#     theta ~ lo + (hi - lo) Beta(a, b),   a = kappa (m - lo)/(hi - lo),   b = kappa - a
# with hard bounds [lo, hi] (never violated, no clipping), mean m (m = 1 keeps the scenarios centred on the base case) and a
# concentration kappa (how uncertain: Var theta = (hi-lo)^2 mu(1-mu)/(kappa+1), mu = (m-lo)/(hi-lo); kappa -> inf deterministic,
# kappa = 2 with m centred = Uniform(lo, hi), kappa < 2 U-shaped).
#
#     quantity   applies to                                   rule                                        base theta
#     "profile"  reference_load_capacity_factor rho_t         rho_t ^ theta (keeps 0 and 1 fixed, no clip) 1   (U2, the base study)
#     "cL"       linear_disutility_DKK_per_kWh                value x theta                               1   (U1, a variation)
#
# The daily minimum is not a separate quantity: in every scenario min_total_energy_per_day_kWh = max_load x sum_t rho_t^theta,
# the daily reference energy of that scenario (deterministic given the profile).
# The hourly bounds follow the perturbed reference as well: l_t in [max(0, l_ref_t - D), min(L_max, l_ref_t + D)] with the
# catalogue's max_hourly_deviation_kWh D (recompute them from the scenario's reference_load_capacity_factor when you load it).
#
# Hyper-parameters per quantity (and, optionally, per consumer): n_scenarios, lo, hi, mean, kappa. n_scenarios = 1 means
# "not uncertain in this experiment": the single scenario is the base value (theta = 1) for every consumer.
# ----------------------------------------------------------------------------------------------------------------------
def _catalogue_by_id(app: dict) -> dict:
    out = {}
    for group, key in (("DERs", "DER_id"), ("loads", "load_id"), ("storages", "storage_id"), ("heat_pumps", "heat_pump_id")):
        for e in app.get(group) or []:
            out[e[key]] = e
    return out


QUANTITIES = ("profile", "cL")


@dataclass
class BetaRule:
    """Hyper-parameters of one scaling parameter: n_scenarios draws of theta ~ lo + (hi - lo) Beta(a, b)."""
    n_scenarios: int = 1
    lo: float = 1.0
    hi: float = 1.0
    mean: float = 1.0
    kappa: float = 2.0

    @property
    def ab(self) -> tuple[float, float]:
        if not (self.lo < self.mean < self.hi):
            raise ValueError(f"the mean must lie strictly inside (lo, hi): lo={self.lo}, mean={self.mean}, hi={self.hi}")
        a = self.kappa * (self.mean - self.lo) / (self.hi - self.lo)
        return a, self.kappa - a

    @property
    def variance(self) -> float:
        mu = (self.mean - self.lo) / (self.hi - self.lo)
        return (self.hi - self.lo) ** 2 * mu * (1 - mu) / (self.kappa + 1)

    def draw(self, rng: np.random.Generator, base: float = 1.0) -> np.ndarray:
        """n_scenarios values of theta; one scenario = the base value (not uncertain)."""
        if self.n_scenarios <= 1:
            return np.array([base])
        a, b = self.ab
        return self.lo + (self.hi - self.lo) * rng.beta(a, b, size=self.n_scenarios)


def rules_to_spec(rules: dict, consumers: list[str]) -> dict:
    """Build the specification {quantity: {consumer_id: BetaRule}} from preference_uncertainty_params.json:
    the entry of each quantity applies to every consumer unless rules["per_consumer"][quantity][consumer_id] overrides it.
    Students tune the specification in code: spec["cL"]["C1"] = BetaRule(n_scenarios=10, lo=0.5, hi=2.0, mean=1.0, kappa=4.0)."""
    spec = {}
    for q in QUANTITIES:
        r = rules["quantities"][q]
        keys = ("n_scenarios", "lo", "hi", "mean", "kappa")
        spec[q] = {}
        for cid in consumers:
            over = rules.get("per_consumer", {}).get(q, {}).get(cid, {})
            spec[q][cid] = BetaRule(**{k: over.get(k, r[k]) for k in keys})
    return spec


def set_rule(spec: dict, quantity: str, consumer_id: str | None = None, **kw) -> dict:
    """Change hyper-parameters of one quantity for one consumer (or for all consumers when consumer_id is None)."""
    for cid, rule in spec[quantity].items():
        if consumer_id is None or cid == consumer_id:
            for k, v in kw.items():
                setattr(rule, k, v)
    return spec


def draw_scaling_scenarios(spec: dict, seed: int) -> dict:
    """Independent scenarios of the scaling parameters, per quantity and per consumer:
    {quantity: {consumer_id: array of theta (length n_scenarios of that rule)}}.
    Every (quantity, consumer) pair has its own random stream (spawned from the seed), so changing the number of scenarios
    of one pair does not change the draws of the others."""
    pairs = [(q, cid) for q in QUANTITIES for cid in spec[q]]
    streams = np.random.SeedSequence(seed).spawn(len(pairs))
    out = {q: {} for q in QUANTITIES}
    for (q, cid), ss in zip(pairs, streams):
        out[q][cid] = spec[q][cid].draw(np.random.default_rng(ss))
    return out


def apply_scaling(load: dict, theta: dict) -> dict:
    """The appliance dict of one load under the scaling parameters theta = {quantity: value}: cL scaled, the profile raised to
    the power theta, and the daily minimum set to the daily reference energy of the resulting profile."""
    a = copy.deepcopy(load)
    # cL: the disutility slope (also applied to a quadratic or linear-utility coefficient if a load has one)
    for key in ("linear_disutility_DKK_per_kWh", "quadratic_disutility_DKK_per_kWh2", "consumption_utility_DKK_per_kWh"):
        if a.get(key) is not None:
            a[key] = a[key] * theta.get("cL", 1.0)
    # profile: rho_t ^ gamma (0 and 1 are fixed points; stays in [0, 1])
    if a.get("reference_load_capacity_factor") is not None:
        rho = np.array(a["reference_load_capacity_factor"], dtype=float) ** theta.get("profile", 1.0)
        a["reference_load_capacity_factor"] = rho.tolist()
        # the daily minimum follows the profile: the daily reference energy of THIS scenario
        if a.get("min_total_energy_per_day_kWh") is not None:
            a["min_total_energy_per_day_kWh"] = float(a["max_load_KWh_per_hour"] * rho.sum())
    return a


def merge_scenarios(app: dict, community: dict, scaling: dict, mode: str = "aligned", n_merged: int | None = None,
                    seed: int | None = None) -> list[dict]:
    """Combine the independent scaling scenarios into joint scenarios of the community, each with the index of the scenario
    it took from every (quantity, consumer):
      "aligned": joint scenario s takes scenario s of every pair (a pair with one scenario contributes its base value); every
                 uncertain pair must then have the same number of scenarios n, and n joint scenarios result (equiprobable);
      "random":  n_merged joint scenarios, each picking an independent uniformly random index per pair (equiprobable);
      "product": every combination of indices (the full product - small cases only).
    Each joint scenario: {"id", "probability", "index": {(quantity, consumer_id): k}, "theta": {quantity: {consumer_id: value}},
                          "members": {consumer_id: {appliance_id: appliance dict}}}."""
    cat = _catalogue_by_id(app)
    ids = [m["consumer_id"] for m in community["consumers"]]
    pairs = [(q, cid) for q in QUANTITIES for cid in ids]
    sizes = [len(scaling[q][cid]) for q, cid in pairs]
    if mode == "aligned":
        ns = {n for n in sizes if n > 1} or {1}
        if len(ns) > 1:
            raise ValueError(f"aligned merging needs the same number of scenarios for every uncertain pair; found {sorted(ns)}")
        n = ns.pop()
        index_sets = [{p: (k if sizes[i] > 1 else 0) for i, p in enumerate(pairs)} for k in range(n)]
    elif mode == "random":
        if n_merged is None:
            raise ValueError("random merging needs n_merged")
        rng = np.random.default_rng(seed)
        index_sets = [{p: int(rng.integers(sizes[i])) for i, p in enumerate(pairs)} for _ in range(n_merged)]
    elif mode == "product":
        import itertools
        index_sets = [dict(zip(pairs, combo)) for combo in itertools.product(*[range(n) for n in sizes])]
    else:
        raise ValueError(mode)
    out = []
    for s, idx in enumerate(index_sets):
        theta = {q: {cid: float(scaling[q][cid][idx[(q, cid)]]) for cid in ids} for q in QUANTITIES}
        members = {}
        for m in community["consumers"]:
            cid = m["consumer_id"]
            members[cid] = {aid: (apply_scaling(cat[aid], {q: theta[q][cid] for q in QUANTITIES}) if "load_id" in cat[aid] else copy.deepcopy(cat[aid]))
                            for aid in m["list_appliances"]}
        out.append({"id": s, "probability": 1.0 / len(index_sets), "index": idx, "theta": theta, "members": members})
    return out


def expected_preference_scenario(app: dict, community: dict) -> dict:
    """consumer_id -> {appliance_id -> appliance dict} with the catalogue parameters unchanged (theta = 1 everywhere)."""
    cat = _catalogue_by_id(app)
    return {m["consumer_id"]: {a: copy.deepcopy(cat[a]) for a in m["list_appliances"]} for m in community["consumers"]}


def draw_preference_scenarios(app: dict, community: dict, rules: dict, n: int, seed: int, quantities=("profile",),
                              spec: dict | None = None, mode: str = "aligned") -> list[dict]:
    """One-call version: n joint scenarios in which the listed quantities are uncertain (with the hyper-parameters of the
    rules file, or of `spec` if given) and the others are at their base value. Returns the joint scenarios of
    merge_scenarios(); the independent scaling draws are in scenario["theta"] and the indices in scenario["index"]."""
    ids = [m["consumer_id"] for m in community["consumers"]]
    spec = spec or rules_to_spec(rules, ids)
    for q in QUANTITIES:
        set_rule(spec, q, None, n_scenarios=(n if q in quantities else 1))
    scaling = draw_scaling_scenarios(spec, seed)
    return merge_scenarios(app, community, scaling, mode=mode, n_merged=n, seed=seed)


if __name__ == "__main__":
    app = load_json("appliance_params.json"); com = load_json("params_community_base.json")
    pref = load_json("preference_uncertainty_params.json")
    week = load_json("weekly_profiles.json"); bus = load_json("bus_params.json")
    app3, bus3 = day_data(app, bus, week, "d3")
    print("Data: day d3 ->", week["days"]["d3"]["name"], "| price mean", round(sum(bus3["energy_price_DKK_per_kWh"]) / 24, 3),
          "| PV ratio at 12h", app3["DERs"][0]["hourly_profile_ratio"][12])

    # Question 3: the reference profiles uncertain (5 scenarios, aligned; the daily minimum follows), cL at its base value
    scen = draw_preference_scenarios(app, com, pref, n=5, seed=1, quantities=("profile",))
    s0 = scen[0]; l0 = s0["members"]["C1"]["FL_L1"]
    print("Q3: 5 joint scenarios; member C1 in scenario 0 -> theta", {q: round(s0["theta"][q]["C1"], 3) for q in QUANTITIES},
          "| c_L =", round(l0["linear_disutility_DKK_per_kWh"], 3), "reference energy =", round(3 * sum(l0["reference_load_capacity_factor"]), 2),
          "E_min =", round(l0["min_total_energy_per_day_kWh"], 2))
    spec = rules_to_spec(pref, [m["consumer_id"] for m in com["consumers"]])
    set_rule(spec, "profile", None, n_scenarios=3, lo=0.9, hi=1.1, mean=1.0, kappa=4.0)    # tune one quantity for all consumers
    set_rule(spec, "cL", "C4", n_scenarios=3, lo=0.5, hi=2.0, mean=1.3, kappa=2.0)         # ... or for one consumer (U1 as a variation)
    sc = draw_scaling_scenarios(spec, seed=7)
    joint = merge_scenarios(app, com, sc, mode="random", n_merged=4, seed=7)
    print("    tuned spec, random merging:", len(joint), "joint scenarios; indices taken by scenario 0 for C4:",
          {q: joint[0]["index"][(q, "C4")] for q in QUANTITIES}, "| E_min/ref energy of C1:",
          round(joint[0]["members"]["C1"]["FL_L1"]["min_total_energy_per_day_kWh"] / (3 * sum(joint[0]["members"]["C1"]["FL_L1"]["reference_load_capacity_factor"])), 3))
