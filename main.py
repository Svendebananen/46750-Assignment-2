"""Entry point of the Assignment 2 starter package: load a community and summarize it.

    python main.py                                   # base case (16 members), day d1
    python main.py --community my_mix                # your own data/params_community_my_mix.json
    python main.py --day d4                          # another day of the week (and that day's cap)
    python main.py --members                         # also print every member's Assignment 1 summary
    python main.py --demo                            # also run the scenario-generation demonstration (Question 3)
    python main.py --R                               # solve the 16-member reference case
    python main.py --I                               # solve the 16-member individual-contract case
    python main.py --C                               # solve the centralized-dispatch community-contract case
    python main.py --q1                              # all --Q1: run --q1-base, --q1-sweep and --q1-settlement (Question 1)
    python main.py --q1-base                         # comparison of R, I and C on the base case (Querstion 1)
    python main.py --q1-sweep                        # sweep of contract paramters (beta^DSO and phi) for I and C (Question 1)
    python main.py --q1-settlement                   # settelement of C and manager's budget (Question 1)

The R, I, and C model runs are implemented in ``src.model`` and write their results to ``results/``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.data_loader import load_community, list_communities
from src.model import CommunityModel, ContractMode, DynamicPricingCommunityModel, internal_price_of_C
from src.plotting import plot_member_benefits, plot_results
from src import analysis_q1 as q1
from src.plotting import plot_gain_map, plot_member_benefits, plot_member_gains, plot_results

RESULTS_DIR = Path(__file__).resolve().parent / "results"

def run_q1_base(com, end_value: float, out: Path) -> None:
    """Comparison for the contract types (models R, I and C on the base case)"""
    table, gains = q1.base_comparison(com, end_value)
    table.round(2).to_csv(out / "q1_table3_base_case.csv")
    print("\nTable 3 - base case:\n", table.round(2).to_string())
    print("Gains (DKK):", {k: round(v, 2) for k, v in gains.items()})


def run_q1_sweep(com, end_value: float, out: Path, betas, phis) -> None:
    """Sensitivity sweep of I and C over the discount beta^DSO and the cap factor phi"""
    grid = q1.contract_sweep(com, end_value, betas, phis)
    grid.to_csv(out / "q1_sweep_beta_cap.csv", index=False)
    print("\nSmallest value of pooling G_C,I over the grid:", round(grid["pooling_value_C_vs_I"].min(), 2), "DKK")
    print(f"  {plot_gain_map(grid, out)}")


def run_q1_settlement(com, end_value: float, out: Path) -> None:
    """Individual rationality of the dual-price settlement of C and the manager's budget."""
    members, hourly, numbers = q1.settlement_study(com, end_value)
    members.round(4).to_csv(out / "q1_settlement_members.csv")
    hourly.round(4).to_csv(out / "q1_settlement_hourly.csv")
    print("\nSettlement of C:", {k: (round(v, 2) if not isinstance(v, tuple) else tuple(round(x, 2) for x in v))
                                  for k, v in numbers.items()})
    for key in ("assets", "slope"):
        print(f"\nGain vs R by {key} (DKK/day):\n", members.groupby(key)["gain_vs_R"].sum().round(2).to_string())
    print(f"  {plot_member_gains(members, out)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--community", default="base", help=f"base (shipped) or your own params_community_<name>.json; available: {list_communities()}")
    ap.add_argument("--day", default="d1", help="d1 (base case) ... d7")
    ap.add_argument("--members", action="store_true", help="print the Assignment 1 summary of every member")
    ap.add_argument("--demo", action="store_true", help="run the scenario-generation demonstration")
    ap.add_argument("--R", action="store_true", help="run the 16-member reference case without a contract")
    ap.add_argument("--I", action="store_true", help="run the 16-member individual-contract case")
    ap.add_argument("--C", action="store_true", help="run the centralized-dispatch community-contract case")
    ap.add_argument("--battery-end-value", type=float, default=0.0, help="economic value in DKK/kWh of battery energy at the end of the day")
    ap.add_argument("--q1", action="store_true", help="Question 1: run --q1-base, --q1-sweep and --q1-settlement")
    ap.add_argument("--q1-base", action="store_true", help="Section 1.4: Table 3 and the gains of R, I and C")
    ap.add_argument("--q1-sweep", action="store_true", help="Section 1.5: gain over beta and cap factor (Figure 1)")
    ap.add_argument("--q1-settlement", action="store_true", help="Sections 1.6-1.7: settlement at lambda_t and manager's budget (Figure 2)")
    ap.add_argument("--betas", type=float, nargs="*", default=[0.0, 0.25, 0.5, 0.75, 1.0], help="discounts of the sweep")
    ap.add_argument("--phis", type=float, nargs="*", default=[0.7, 0.8, 0.9, 1.0, 1.15, 1.3, 1.6, 2.0],
                    help="cap factors of the sweep (1.3 = base case)")
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
    out = RESULTS_DIR / args.community
    if args.q1 or args.q1_base:
        run_q1_base(com, args.battery_end_value, out)
    if args.q1 or args.q1_sweep:
        run_q1_sweep(com, args.battery_end_value, out, args.betas, args.phis)
    if args.q1 or args.q1_settlement:
        run_q1_settlement(com, args.battery_end_value, out)


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
