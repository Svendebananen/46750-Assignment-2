"""Entry point of the Assignment 2 starter package: load a community and summarize it.

    python main.py                                   # base case (16 members), day d1
    python main.py --community my_mix                # your own data/params_community_my_mix.json
    python main.py --day d4                          # another day of the week (and that day's cap)
    python main.py --members                         # also print every member's Assignment 1 summary
    python main.py --demo                            # also run the scenario-generation demonstration (Question 3)
    python main.py --R                               # solve the 16-member reference case
    python main.py --I                               # solve the 16-member individual-contract case
    python main.py --C                               # solve the centralized-dispatch community-contract case

The R, I, and C model runs are implemented in ``src.model`` and write their results to ``results/``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.data_loader import load_community, list_communities
from src.model import CommunityModel, ContractMode, DynamicPricingCommunityModel, internal_price_of_C
from src.plotting import (plot_dp_benefits_by_member, plot_dp_cap_sweep, plot_dp_compare_prices,
                          plot_dp_cost_of_requirements, plot_dp_prices, plot_dp_shift_heatmap, plot_member_benefits,
                          plot_results)

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--community", default="base", help=f"base (shipped) or your own params_community_<name>.json; available: {list_communities()}")
    ap.add_argument("--day", default="d1", help="d1 (base case) ... d7")
    ap.add_argument("--members", action="store_true", help="print the Assignment 1 summary of every member")
    ap.add_argument("--demo", action="store_true", help="run the scenario-generation demonstration")
    ap.add_argument("--R", action="store_true", help="run the 16-member reference case without a contract")
    ap.add_argument("--I", action="store_true", help="run the 16-member individual-contract case")
    ap.add_argument("--C", action="store_true", help="run the centralized-dispatch community-contract case")
    ap.add_argument("--battery-end-value", type=float, default=0.0,
                    help="economic value in DKK/kWh of battery energy at the end of the day")
    ap.add_argument("--DP", action="store_true", help="run DP and compare it with C")
    ap.add_argument("--sos1", action="store_true", help="DP: SOS1 instead of big-M")
    ap.add_argument("--reg", type=float, default=1e-3, help="DP: weight of the price regularization towards C's duals")
    ap.add_argument("--time-limit", type=float, default=None)
    ap.add_argument("--bb", action="store_true", help="DP: budget balance")
    ap.add_argument("--ir", action="store_true", help="DP: soft individual rationality against R (adds budget balance)")
    ap.add_argument("--ir-penalty", type=float, default=100.0, help="DP: kappa, DKK per DKK of IR slack")
    ap.add_argument("--tight-M", action="store_true", help="DP, big-M: one M^P and M^D per complementarity set")
    ap.add_argument("--M-scale", type=float, default=1.0, help="DP, --tight-M: safety factor on the per-set Ms")
    ap.add_argument("--mip-focus", type=int, default=0, choices=(0, 1, 2, 3),
                    help="DP: Gurobi MIPFocus (1 = look for good solutions first, 3 = work on the bound)")
    ap.add_argument("--ir-bench", choices=("R", "I", "best"), default="best",
                    help="DP: IR benchmark per member: R, I, or the better of the two (default, as in A5)")
    ap.add_argument("--cap-f", type=float, nargs="+", default=[1.3],
                    help="DP: cap tightness f' (cap x f'/1.3; 1.3 = base). Several values = the sensitivity study")
    args = ap.parse_args()

    com = load_community(args.community, day=args.day)
    print(com.summary(), "\n")
    if args.members:
        for m in com.members:
            print(m.summary(), "\n")
    print("Cap profile (kW):", " ".join(f"{x:.1f}" for x in com.contract.capacity_limit_kW))
    ic = com.contract.individual_capacity_limit_kW
    print("Individual caps in model I (kW), hour 0 / hour 11:",
          "  ".join(f"{cid} ({com.member(cid).archetype}) {ic[cid][0]:.2f}/{ic[cid][11]:.2f}" for cid in ("C1", "C1_PVB", "C4", "C4_PVB") if cid in com.member_ids))
    (RESULTS_DIR / args.community).mkdir(parents=True, exist_ok=True)
    if args.demo:
        import runpy
        runpy.run_module("src.scenario_generation", run_name="__main__")
    if args.DP:
        run_dp(com, args)
    if args.R or args.I or args.C:
        mode = (
            ContractMode.COMMUNITY
            if args.C
            else ContractMode.INDIVIDUAL
            if args.I
            else ContractMode.NONE
        )
        result = CommunityModel(
            com,
            contract_mode=mode,
            battery_end_soc_value_DKK_per_kWh=args.battery_end_value,
        ).build().solve()
        print(result)
        result.save(RESULTS_DIR / args.community, tag=mode.value)
        plot_paths = plot_results(result, com.contract, com.energy_price, RESULTS_DIR / args.community)
        print("Plots written:")
        for path in plot_paths:
            print(f"  {path}")
        if mode != ContractMode.NONE:
            reference = CommunityModel(
                com,
                contract_mode=ContractMode.NONE,
                battery_end_soc_value_DKK_per_kWh=args.battery_end_value,
            ).build().solve()
            benefit_path = plot_member_benefits(
                reference,
                result,
                com.members,
                RESULTS_DIR / args.community,
            )
            print(f"  {benefit_path}")

def solve_dp(com, args, lam_C, label: str, budget_balance: bool = False, ir_benchmark=None):
    """Build and solve one DP variant and print the course's validation checks. Returns None if there is no solution."""
    dp = DynamicPricingCommunityModel(
        com, method="SOS1" if args.sos1 else "bigM",
        battery_end_soc_value_DKK_per_kWh=args.battery_end_value,
        reg_weight=args.reg, reg_reference=lam_C if args.reg > 0 else None,
        time_limit=args.time_limit, budget_balance=budget_balance,
        ir_benchmark=ir_benchmark, ir_penalty=args.ir_penalty,
        tight_M=args.tight_M, M_scale=args.M_scale, mip_focus=args.mip_focus, verbose=True).build()
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
    r.meta["member_shift"] = {d.consumer_id: [dp.var["load"][i, t].X - d.reference_load[t] for t in dp.T]
                              for i, d in enumerate(dp.members)}             # L_it - L_it^ref, for the heat map
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
    table = pd.DataFrame(rows)
    table.insert(2, "dW vs C", table["welfare cost"] - table.loc[0, "welfare cost"])
    for col in ("energy", "tariffs", "penalty", "PV + discomfort"):
        table[f"d {col}"] = table[col] - table.loc[0, col]
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
        paths = [plot_dp_cost_of_requirements(table, out, tag=stem),
                 plot_dp_benefits_by_member({"C (dual prices)": mC, **mtabs}, args.ir_bench, out, tag=stem)]
        for label, r in runs.items():
            paths += plot_results(r, com.contract, com.energy_price, out, tag=tags[label])
            paths.append(plot_dp_prices(r, lam_C, com, out, tag=tags[label]))
        if len(runs) > 1:
            paths.append(plot_dp_compare_prices(runs, lam_C, com, out, tag=base_tag + "_compare"))
            paths.append(plot_dp_shift_heatmap(shifts, {m.consumer_id: asset_type(m) for m in com.members},
                                           com.contract.capacity_limit_kW,
                                           {STEP_NAMES[l]: r.hourly["import"].to_numpy() for l, r in runs.items()},
                                           out, base="DP", compare="DP^BB,IR" if "DP^BB,IR" in shifts else None,
                                           tag=stem))
        print("Plots written:")
        for path in paths:
            print(f"  {path}")
    return rows


def run_dp(com, args) -> None:
    """Question 2. One --cap-f value: the steps on that contract. Several: the same for every value + the sweep."""
    import pandas as pd
    out = RESULTS_DIR / args.community
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
    print("  ", plot_dp_cap_sweep(sweep, out, tag=f"{base_tag}_q2"))

if __name__ == "__main__":
    main()
