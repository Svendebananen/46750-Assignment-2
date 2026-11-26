"""Question 2 dynamic-pricing experiment, reporting, and sensitivity analysis."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .data_loader import CommunityData
from .dynamic_pricing_model import DynamicPricingCommunityModel, internal_price_of_C
from .dynamic_pricing_plotting import (
    DynamicPricingPlotter,
)
from .model import CommunityModel, ContractMode
from .plotting import plot_results


def solve_dp(com, args, lam_C, label: str, budget_balance: bool = False, ir_benchmark=None):
    """Build and solve one DP variant and print the course's validation checks. Returns None if there is no solution."""
    dp = DynamicPricingCommunityModel(
        com, method="SOS1" if args.sos1 else "bigM",
        battery_end_soc_value_DKK_per_kWh=args.battery_end_value,
        reg_weight=args.reg, reg_reference=lam_C if args.reg > 0 else None,
        time_limit=args.time_limit, budget_balance=budget_balance,
        ir_benchmark=ir_benchmark, ir_penalty=args.ir_penalty,
        tight_M=args.tight_M, M_scale=args.M_scale, mip_focus=args.mip_focus,
        threads=args.threads, verbose=True).build()
    try:
        r = dp.solve()
    except RuntimeError as err:
        print(f"\n--- DP {label}: NO SOLUTION ({err})")
        return None
    r.hourly["excess"] = [dp.var["excess"][t].X for t in dp.T]       # e_t of the model (for the budget-balance check)
    r.hourly["lambda_C"] = lam_C
    r.meta["runtime_s"] = dp.m.Runtime
    r.meta["mip_bound"] = dp.m.ObjBound
    r.meta["ir_penalty"] = args.ir_penalty if ir_benchmark is not None else None
    member_shift = {}
    for i, member in enumerate(dp.members):
        if member.reference_load is None:
            raise ValueError(f"{member.question} does not define reference_load.")
        member_shift[member.consumer_id] = [
            dp.var["load"][i, t].X - member.reference_load[t] for t in dp.T
        ]
    r.meta["member_shift"] = member_shift
    v, x = r.meta["validation"], r.hourly["posted_price"].to_numpy()
    print(f"\n--- DP {label}  ({r.status}, MIP gap {r.meta['mip_gap']:.2e}, {r.meta['runtime_s']:.0f} s)")
    print(f"DP welfare cost        : {r.meta['welfare_cost']:.4f} DKK")
    print(f"max |x_t - lambda_t^C| : {max(abs(x[t] - lam_C[t]) for t in range(len(x))):.4f}")
    print(f"max s*mu               : {v['max_complementarity_residual']:.2e}")
    print(f"max s/M^P, max mu/M^D  : {v['max_s_over_MP']:.3f}, {v['max_mu_over_MD']:.3f}"
          f"   (sets with mu at M^D: {', '.join(v['sets_at_M_D']) or 'none'})")
    print(f"hours at a price bound : {v['hours_price_at_bound']}")
    print(f"indifferent pairs      : {len(v['indifferent_pairs'])}")
    print(f"manager surplus        : {r.meta['manager_surplus']:+.4f} DKK")
    if "max_strong_duality_gap" in v:
        print(f"max |D_i - cost_i|     : {v['max_strong_duality_gap']:.2e}   (strong duality check)")
    if "individual_rationality" in r.meta:
        blocking = {cid: d["slack"] for cid, d in r.meta["individual_rationality"].items() if d["slack"] > 1e-6}
        print("blocking members       : " + (", ".join(f"{c} ({z:.2f} DKK)" for c, z in blocking.items()) or "none"))
        print(f"total compensation     : {sum(blocking.values()):.3f} DKK")
    return r


STEP_NAMES = {"cap only": "DP", "with budget balance": "DP^BB", "with budget balance + IR": "DP^BB,IR"}


def asset_type(m) -> str:
    return ("PV and battery" if m.has_pv and m.has_battery else "PV" if m.has_pv
            else "Battery" if m.has_battery else "No asset")


def member_order(members) -> list:
    """Members grouped by asset type, then by load slope: the order of every member table and figure."""
    rank = {"No asset": 0, "PV": 1, "Battery": 2, "PV and battery": 3}
    return sorted(members, key=lambda m: (rank[asset_type(m)], m.linear_disutility, m.consumer_id))


def ir_benchmark_costs(R, I, rule: str) -> dict:
    """C_i^bench: the member's cost without the community, R, I or the better of the two."""
    cR, cI = R.meta["member_operating_costs"], I.meta["member_operating_costs"]
    return {cid: cR[cid] if rule == "R" else cI[cid] if rule == "I" else min(cR[cid], cI[cid]) for cid in cR}


def q2_metrics(r, com, prices) -> dict:
    """Community metrics of one run (C or a DP step), from its hourly flows. DKK/day, kWh/day, kW."""
    import numpy as np
    h, c = r.hourly, com.contract
    p = np.asarray(com.energy_price)
    imp, exp = h["import"].to_numpy(), h["export"].to_numpy()
    cap = np.asarray(c.capacity_limit_kW)
    beta, alpha = c.tariff_discount_ratio, c.penalty_DKK_per_kWh
    ti, te = com.import_tariff, com.export_tariff
    over = np.maximum(imp - cap, 0.0)
    energy = float(p @ (imp - exp))
    tariffs = float((1 - beta) * (ti * imp.sum() + te * exp.sum()))
    penalty = float((alpha + beta * ti) * over.sum())
    meter = energy + tariffs + penalty
    welfare = float(r.meta.get("welfare_cost", r.objective))
    payments = float(np.asarray(prices) @ (imp - exp))
    return {"welfare cost": welfare, "energy": energy, "tariffs": tariffs, "penalty": penalty,
            "PV + discomfort": welfare - meter, "exceedance kWh": float(over.sum()),
            "hours above cap": int((over > 1e-6).sum()), "hours at cap": int((abs(imp - cap) <= 1e-3).sum()),
            "peak kW": float(imp.max()), "payments": payments, "meter bill": meter, "manager surplus": payments - meter}


def member_table(r, R, I, bench: dict, members) -> "pd.DataFrame":
    """One row per member: costs in R, I, the benchmark and the run, and the benefits (> 0 = better off)."""
    import pandas as pd
    cR, cI, c = R.meta["member_operating_costs"], I.meta["member_operating_costs"], r.meta["member_operating_costs"]
    ir = r.meta.get("individual_rationality", {})
    rows = [{"member": m.consumer_id, "assets": asset_type(m), "c_L": m.linear_disutility,
             "cost R": cR[m.consumer_id], "cost I": cI[m.consumer_id], "benchmark": bench[m.consumer_id],
             "cost": c[m.consumer_id], "benefit vs R": cR[m.consumer_id] - c[m.consumer_id],
             "benefit vs I": cI[m.consumer_id] - c[m.consumer_id], "benefit": bench[m.consumer_id] - c[m.consumer_id],
             "zeta": ir.get(m.consumer_id, {}).get("slack", float("nan"))} for m in member_order(members)]
    return pd.DataFrame(rows)


def validation_row(r, lam_C) -> dict:
    """Validation checks of one DP step, incl. the budget-balance check: e = (import - cap)^+, no import + export."""
    import numpy as np
    v, h = r.meta["validation"], r.hourly
    imp, exp = h["import"].to_numpy(), h["export"].to_numpy()
    return {"status": r.status, "MIP gap": r.meta["mip_gap"], "time s": r.meta.get("runtime_s", float("nan")),
            "binaries": r.meta["binaries"],
            "max |x - lambda_C|": float(np.max(np.abs(h["posted_price"].to_numpy() - np.asarray(lam_C)))),
            "max s*mu": v["max_complementarity_residual"],
            "max |D_i - cost_i|": v.get("max_strong_duality_gap", float("nan")),
            "indifferent pairs": len(v["indifferent_pairs"]),
            "hours at price bound": " ".join(map(str, v["hours_price_at_bound"])) or "-",
            "max |e - (imp - cap)+|": float(np.max(np.abs(h["excess"].to_numpy() - np.maximum(
                imp - np.asarray(r.meta["cap"]), 0.0)))) if "excess" in h else float("nan"),
            "max min(imp, exp)": float(np.max(np.minimum(imp, exp)))}


def write_latex(df, path, caption: str, label: str, digits: int = 2) -> None:
    """A plain LaTeX table (booktabs)."""
    names = {"C (ref.)": r"$\mathcal C$ (ref.)", "DP": r"$\mathcal{DP}$", "DP^BB": r"$\mathcal{DP}^{BB}$",
             "DP^BB,IR": r"$\mathcal{DP}^{BB,IR}$"}

    def fmt(v):
        if isinstance(v, float):
            if v != v:
                return "--"
            if abs(v) < 1e-6:
                return f"{0:.{digits}f}"
            return f"{v:.{digits}f}" if abs(v) >= 1e-3 else f"{v:.1e}"
        return names.get(str(v), str(v).replace("_", r"\_").replace("%", r"\%"))
    cols = list(df.columns)
    lines = [r"\begin{table}[ht]", r"\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
             r"\begin{tabular}{l" + "r" * (len(cols) - 1) + "}", r"\toprule",
             " & ".join(fmt(c) for c in cols) + r" \\", r"\midrule"]
    lines += [" & ".join(fmt(v) for v in row) + r" \\" for row in df.itertuples(index=False)]
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def scaled_cap(com, f: float):
    """Cap tightness (Table 11): the community cap and the individual caps multiplied by f'/1.3 (1.3 = base)."""
    import dataclasses
    s = f / 1.3
    c = com.contract
    return dataclasses.replace(com, contract=dataclasses.replace(
        c, capacity_limit_kW=c.capacity_limit_kW * s,
        individual_capacity_limit_kW={k: v * s for k, v in c.individual_capacity_limit_kW.items()}))


def dual_settlement_costs(C, lam_C) -> dict:
    """Each member's cost in C when settled at C's internal prices lambda_t^C (the settlement of Question 1)."""
    costs = {}
    for i, d in enumerate(C.members):
        cost = 0.0
        for t in C.T:
            pv = C._member_var("pv", i, t)
            cost += (lam_C[t] * C.var["net_consumption"][i, t].X + d.pv_marginal_cost * (pv.X if hasattr(pv, "X") else 0.0)
                     + d.linear_disutility * C.var["deviation"][i, t].X)
        if d.has_battery:
            cost -= C.terminal_value * C.var["soc"][i][max(C.T)].X
        costs[d.consumer_id] = float(cost)
    return costs


def run_dp_case(com, args, out, plots: bool = True) -> list[dict]:
    """C, R, I, then the DP steps for one contract: tables T1, T2, member tables, figures."""
    import json
    import pandas as pd
    out.mkdir(parents=True, exist_ok=True)
    C = CommunityModel(com, contract_mode=ContractMode.COMMUNITY,
                       battery_end_soc_value_DKK_per_kWh=args.battery_end_value).build()
    rc = C.solve()
    lam_C = internal_price_of_C(C)
    rc.hourly["excess"] = [C.var["excess"][t].X for t in C.T]
    rc.meta["member_operating_costs_meter_split"] = rc.meta["member_operating_costs"]
    rc.meta["member_operating_costs"] = dual_settlement_costs(C, lam_C)     # C settled at its internal prices (Q1)
    R = CommunityModel(com, contract_mode=ContractMode.NONE,
                       battery_end_soc_value_DKK_per_kWh=args.battery_end_value).build().solve()
    I = CommunityModel(com, contract_mode=ContractMode.INDIVIDUAL,
                       battery_end_soc_value_DKK_per_kWh=args.battery_end_value).build().solve()
    bench = ir_benchmark_costs(R, I, args.ir_bench)
    print(f"C  welfare cost        : {rc.objective:.4f} DKK   (IR benchmark: {args.ir_bench})")

    base_tag = "dynamic_sos1" if args.sos1 else ("dynamic_tightM" if args.tight_M else "dynamic")
    steps = [("cap only", "", dict())]
    if args.bb or args.ir:
        steps.append(("with budget balance", "_bb", dict(budget_balance=True)))
    if args.ir:
        steps.append(("with budget balance + IR", "_bb_ir", dict(budget_balance=True, ir_benchmark=bench)))
    runs, tags = {}, {}
    for label, suffix, kw in steps:
        r = solve_dp(com, args, lam_C, label, **kw)
        if r is not None:
            r.meta["cap"] = list(com.contract.capacity_limit_kW)
            runs[label], tags[label] = r, base_tag + suffix

    rows, mtabs = [], {}
    mC = member_table(rc, R, I, bench, com.members)
    rows.append({"model": "C (ref.)", "status": rc.status, **q2_metrics(rc, com, lam_C),
                 "members worse off": int((mC["benefit"] < -1e-6).sum()),
                 "loss of members worse off": float(-mC.loc[mC["benefit"] < -1e-6, "benefit"].sum()),
                 "sum zeta": float("nan"), "MIP gap": 0.0})
    for label, r in runs.items():
        mt = member_table(r, R, I, bench, com.members)
        mtabs[STEP_NAMES[label]] = mt
        worse = mt["benefit"] < -1e-6
        rows.append({"model": STEP_NAMES[label], "status": r.status, **q2_metrics(r, com, r.hourly["posted_price"]),
                     "members worse off": int(worse.sum()), "loss of members worse off": float(-mt.loc[worse, "benefit"].sum()),
                     "sum zeta": float(mt["zeta"].sum()) if mt["zeta"].notna().any() else float("nan"),
                     "MIP gap": r.meta["mip_gap"]})
        mt.to_csv(out / f"{r.question}_{tags[label]}_members.csv", index=False)
        r.save(out, tag=tags[label])
        meta = {k: v for k, v in r.meta.items() if k != "validation"}
        meta["validation"] = {k: v for k, v in r.meta["validation"].items() if k != "indifferent_pairs"}
        meta["validation"]["indifferent_pairs"] = len(r.meta["validation"]["indifferent_pairs"])
        (out / f"{r.question}_{tags[label]}_meta.json").write_text(
            json.dumps(meta, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)), encoding="utf-8")
    table: pd.DataFrame = pd.DataFrame(rows)
    reference_welfare = rows[0]["welfare cost"]
    table.insert(2, "dW vs C", table["welfare cost"].astype(float) - reference_welfare)
    for col in ("energy", "tariffs", "penalty", "PV + discomfort"):
        reference_value = rows[0][col]
        table[f"d {col}"] = table[col].astype(float) - reference_value
    dcols = [c for c in table.columns if c.startswith("d")]           # solver noise (1e-14) is not a cost
    table[dcols] = table[dcols].where(table[dcols].abs() > 1e-6, 0.0)
    val = pd.DataFrame([{"model": STEP_NAMES[l], **validation_row(r, lam_C)} for l, r in runs.items()])
    order = [m.consumer_id for m in member_order(com.members)]
    shifts = {STEP_NAMES[l]: pd.DataFrame(r.meta["member_shift"]).T.loc[order] for l, r in runs.items()}
    for l, r in runs.items():
        shifts[STEP_NAMES[l]].to_csv(out / f"{r.question}_{tags[l]}_member_shift.csv", index_label="member")

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    show = ["model", "status", "welfare cost", "dW vs C", "exceedance kWh", "peak kW", "hours at cap", "payments",
            "meter bill", "manager surplus", "members worse off", "loss of members worse off", "sum zeta", "MIP gap"]
    print("\n--- T2: the steps of Question 2 against C (DKK/day, kWh/day, kW); benefit benchmark =", args.ir_bench)
    print(table[show].round(3).to_string(index=False))
    print("\n--- T1: validation")
    print(val.to_string(index=False))
    for name, mt in mtabs.items():
        print(f"\n--- members, {name} (benefit > 0 = better off than the benchmark)")
        print(mt.round(3).to_string(index=False))

    stem = f"{base_tag}_q2"
    table.to_csv(out / f"{stem}_steps.csv", index=False)
    val.to_csv(out / f"{stem}_validation.csv", index=False)
    write_latex(table[["model", "welfare cost", "dW vs C", "exceedance kWh", "peak kW", "manager surplus",
                       "members worse off", "sum zeta", "MIP gap"]], out / f"{stem}_steps.tex",
                "The steps of Question 2 against $\\mathcal C$ (DKK/day, kWh/day, kW).", "tab:q2_steps")
    write_latex(val.drop(columns=["status"]), out / f"{stem}_validation.tex", "Validation of the DP steps.",
                "tab:q2_validation", digits=3)

    if plots and runs:
        plotter = DynamicPricingPlotter(out)
        paths = [plotter.cost_of_requirements(table, tag=stem),
                 plotter.benefits_by_member({"C (dual prices)": mC, **mtabs}, args.ir_bench, tag=stem)]
        for label, r in runs.items():
            paths += plot_results(r, com.contract, com.energy_price, out, tag=tags[label])
            paths.append(plotter.prices(r, lam_C, com, tag=tags[label]))
        if len(runs) > 1:
            paths.append(plotter.compare_prices(runs, lam_C, com, tag=base_tag + "_compare"))
            paths.append(plotter.shift_heatmap(
                shifts,
                {m.consumer_id: asset_type(m) for m in com.members},
                com.contract.capacity_limit_kW,
                {STEP_NAMES[l]: r.hourly["import"].to_numpy() for l, r in runs.items()},
                base="DP",
                compare="DP^BB,IR" if "DP^BB,IR" in shifts else None,
                tag=stem,
            ))
        print("Plots written:")
        for path in paths:
            print(f"  {path}")
    return rows


def run_dp(com, args, results_dir: Path) -> None:
    """Question 2. One --cap-f value: the steps on that contract. Several: the same for every value + the sweep."""
    import pandas as pd
    out = results_dir / args.community
    if len(args.cap_f) == 1:
        f = args.cap_f[0]
        run_dp_case(scaled_cap(com, f) if f != 1.3 else com, args, out if f == 1.3 else out / f"cap_f{f:.2f}")
        return
    sweep = []
    for f in args.cap_f:
        print(f"\n================ cap tightness f' = {f:.2f} (cap x {f / 1.3:.3f})")
        rows = run_dp_case(scaled_cap(com, f), args, out / f"cap_f{f:.2f}", plots=(f == 1.3))
        sweep += [{"f": f, **row} for row in rows]
    sweep = pd.DataFrame(sweep)
    base_tag = "dynamic_sos1" if args.sos1 else ("dynamic_tightM" if args.tight_M else "dynamic")
    sweep.to_csv(out / f"{base_tag}_q2_cap_sweep.csv", index=False)
    print("\n--- cap sweep")
    print(sweep[["f", "model", "status", "welfare cost", "manager surplus", "exceedance kWh",
                 "members worse off", "sum zeta", "MIP gap"]].round(3).to_string(index=False))
    print("  ", DynamicPricingPlotter(out).cap_sweep(sweep, tag=f"{base_tag}_q2"))


class DynamicPricingExperiment:
    """Run and report Question 2 experiments for one community."""

    def __init__(self, results_dir: Path | str):
        self.results_dir = Path(results_dir)

    def run(self, community: CommunityData, args) -> None:
        run_dp(community, args, self.results_dir)
