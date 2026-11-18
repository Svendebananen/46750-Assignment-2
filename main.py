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
from src.plotting import plot_member_benefits, plot_results

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
    ap.add_argument("--reg", type=float, default=1e-4, help="DP: weight of the price regularization towards C's duals")
    ap.add_argument("--time-limit", type=float, default=None)
    ap.add_argument("--bb", action="store_true", help="DP: budget balance")
    ap.add_argument("--ir", action="store_true", help="DP: soft individual rationality against R (adds budget balance)")
    ap.add_argument("--ir-penalty", type=float, default=100.0, help="DP: kappa, DKK per DKK of IR slack")
    ap.add_argument("--tight-M", action="store_true", help="DP, big-M: one M^P and M^D per complementarity set")
    ap.add_argument("--M-scale", type=float, default=1.0, help="DP, --tight-M: safety factor on the per-set Ms")
    ap.add_argument("--mip-focus", type=int, default=0, choices=(0, 1, 2, 3),
                    help="DP: Gurobi MIPFocus (1 = look for good solutions first, 3 = work on the bound)")
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


def run_dp(com, args) -> None:
    C = CommunityModel(com, contract_mode=ContractMode.COMMUNITY,
                       battery_end_soc_value_DKK_per_kWh=args.battery_end_value).build()
    rc = C.solve()
    lam_C = internal_price_of_C(C)
    bench = None
    if args.ir:                                                     # R = each member alone, no contract
        R = CommunityModel(com, contract_mode=ContractMode.NONE,
                           battery_end_soc_value_DKK_per_kWh=args.battery_end_value).build().solve()
        bench = R.meta["member_operating_costs"]
    dp = DynamicPricingCommunityModel(
        com, method="SOS1" if args.sos1 else "bigM",
        battery_end_soc_value_DKK_per_kWh=args.battery_end_value,
        reg_weight=args.reg, reg_reference=lam_C if args.reg > 0 else None,
        time_limit=args.time_limit, budget_balance=args.bb or args.ir,
        ir_benchmark=bench, ir_penalty=args.ir_penalty,
        tight_M=args.tight_M, M_scale=args.M_scale, mip_focus=args.mip_focus, verbose=True).build()
    r = dp.solve()
    v, x = r.meta["validation"], r.hourly["posted_price"].to_numpy()
    print(f"C  welfare cost : {rc.objective:.4f} DKK")
    print(f"DP welfare cost : {r.meta['welfare_cost']:.4f} DKK  (difference {r.meta['welfare_cost'] - rc.objective:+.4f})")
    print(f"max |x_t - lambda_t^C| : {max(abs(x[t] - lam_C[t]) for t in range(len(x))):.4f}")
    print(f"max s*mu               : {v['max_complementarity_residual']:.2e}")
    print(f"max s/M^P, max mu/M^D  : {v['max_s_over_MP']:.3f}, {v['max_mu_over_MD']:.3f}"
          f"   (sets with mu at M^D: {', '.join(v['sets_at_M_D']) or 'none'})")
    print(f"hours at a price bound : {v['hours_price_at_bound']}")
    print(f"indifferent pairs      : {len(v['indifferent_pairs'])}")
    print(f"manager surplus        : {r.meta['manager_surplus']:+.4f} DKK")
    r.hourly["lambda_C"] = lam_C
    if "max_strong_duality_gap" in v:
        print(f"max |D_i - cost_i|     : {v['max_strong_duality_gap']:.2e}   (strong duality check)")
    print(f"status, MIP gap        : {r.status}, {r.meta['mip_gap']:.2e}")
    if "individual_rationality" in r.meta:
        ir_tab = r.meta["individual_rationality"]
        print(f"{'member':10s} {'cost in R':>10s} {'cost in DP':>11s} {'benefit':>9s} {'slack':>8s}")
        for cid, d in ir_tab.items():
            print(f"{cid:10s} {d['benchmark']:10.3f} {d['cost_DP']:11.3f} {d['benchmark'] - d['cost_DP']:+9.3f} {d['slack']:8.3f}")
        blocking = {cid: d["slack"] for cid, d in ir_tab.items() if d["slack"] > 1e-6}
        print("blocking members       : " + (", ".join(f"{c} ({z:.2f} DKK)" for c, z in blocking.items()) or "none"))
        print(f"total compensation     : {sum(blocking.values()):.3f} DKK")
    tag = "dynamic_sos1" if args.sos1 else ("dynamic_tightM" if args.tight_M else "dynamic")
    if args.bb or args.ir:
        tag += "_bb"
    if args.ir:
        tag += "_ir"
    r.save(RESULTS_DIR / args.community, tag=tag)

if __name__ == "__main__":
    main()
