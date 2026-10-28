"""Entry point of the Assignment 2 starter package: load a community and summarize it.

    python main.py                                   # base case (16 members), day d1
    python main.py --community my_mix                # your own data/params_community_my_mix.json
    python main.py --day d4                          # another day of the week (and that day's cap)
    python main.py --members                         # also print every member's Assignment 1 summary
    python main.py --demo                            # also run the scenario-generation demonstration (Question 3)
    python main.py --R                               # solve the 16-member reference case
    python main.py --I                               # solve the 16-member individual-contract case

The R and I model runs are implemented in ``src.model`` and write their results to ``results/``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.data_loader import load_community, list_communities
from src.model import CommunityModel, ContractMode

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--community", default="base", help=f"base (shipped) or your own params_community_<name>.json; available: {list_communities()}")
    ap.add_argument("--day", default="d1", help="d1 (base case) ... d7")
    ap.add_argument("--members", action="store_true", help="print the Assignment 1 summary of every member")
    ap.add_argument("--demo", action="store_true", help="run the scenario-generation demonstration")
    ap.add_argument("--R", action="store_true", help="run the 16-member reference case without a contract")
    ap.add_argument("--I", action="store_true", help="run the 16-member individual-contract case")
    ap.add_argument("--battery-end-value", type=float, default=0.0,
                    help="economic value in DKK/kWh of battery energy at the end of the day")
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
    if args.R or args.I:
        mode = ContractMode.INDIVIDUAL if args.I else ContractMode.NONE
        result = CommunityModel(
            com,
            contract_mode=mode,
            battery_end_soc_value_DKK_per_kWh=args.battery_end_value,
        ).build().solve()
        print(result)
        result.save(RESULTS_DIR / args.community, tag=mode.value)


if __name__ == "__main__":
    main()
