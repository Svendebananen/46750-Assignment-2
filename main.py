"""Entry point of the Assignment 2 starter package: load a community and summarize it.

    python main.py                                   # base case (16 members), day d1
    python main.py --community my_mix                # your own data/params_community_my_mix.json
    python main.py --day d4                          # another day of the week (and that day's cap)
    python main.py --members                         # also print every member's Assignment 1 summary
    python main.py --demo                            # also run the scenario-generation demonstration (Question 3)
    python main.py --R                               # solve the 16-member reference case
    python main.py --I                               # solve the 16-member individual-contract case
    python main.py --C                               # solve the centralized-dispatch community-contract case
    python main.py --DP                              # run the dynamic-pricing experiment (Question 2)

The R, I, and C model runs are implemented in ``src.model``. Question 2 is implemented in
``src.dynamic_pricing_*``. Results are written to ``results/``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.data_loader import load_community, list_communities
from src.dynamic_pricing_analysis import DynamicPricingExperiment
from src.model import CommunityModel, ContractMode
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
        DynamicPricingExperiment(RESULTS_DIR).run(com, args)
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

if __name__ == "__main__":
    main()
