"""Command-line interface for the three assignment questions and model checks.

Examples: ``python main.py --q1``, ``python main.py --DP --ir``,
``python main.py --q3``, and ``python main.py --test``.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import unittest

from src.data_loader import list_communities, load_community
from src.q1_analysis import Question1Experiment
from src.q2_analysis import DynamicPricingExperiment
from src.model import CommunityModel, ContractMode
from src.plotting import plot_member_benefits, plot_results

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
REPOSITORY_ROOT = Path(__file__).resolve().parent.parent


def run_unit_tests() -> None:
    """Run the fast, solver-free test suite."""
    suite = unittest.defaultTestLoader.discover(
        str(REPOSITORY_ROOT / "tests"),
        top_level_dir=str(REPOSITORY_ROOT),
    )
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--community",
        default="base",
        help=f"base (shipped) or your own params_community_<name>.json; available: {list_communities()}",
    )
    ap.add_argument("--day", default="d1", help="d1 (base case) ... d7")
    ap.add_argument("--members", action="store_true", help="print the Assignment 1 summary of every member")
    ap.add_argument("--demo", action="store_true", help="run the scenario-generation demonstration (Question 3)")
    ap.add_argument("--R", action="store_true", help="run the reference case without a contract")
    ap.add_argument("--I", action="store_true", help="run the individual-contract case")
    ap.add_argument("--C", action="store_true", help="run the centralized-dispatch community-contract case")
    ap.add_argument(
        "--battery-end-value",
        type=float,
        default=0.0,
        help="economic value in DKK/kWh of battery energy at the end of the day",
    )
    ap.add_argument("--final-soc", choices=("initial", "free"), default="initial",
        help="battery end-of-day state of charge: equal to the initial state (as in the reference of the contract file) or free"
    )
    ap.add_argument("--q1", action="store_true", help="run all Question 1 experiments")
    ap.add_argument("--q1-base", action="store_true", help="compare R, I, and C on the base case")
    ap.add_argument("--q1-sweep", action="store_true", help="sweep the contract discount and cap factor")
    ap.add_argument("--q1-settlement", action="store_true", help="analyze the community-contract settlement")
    ap.add_argument(
        "--betas",
        type=float,
        nargs="*",
        default=[0.0, 0.25, 0.5, 0.75, 1.0],
        help="discount values for the Question 1 sweep",
    )
    ap.add_argument(
        "--phis",
        type=float,
        nargs="*",
        default=[0.7, 0.8, 0.9, 1.0, 1.15, 1.3, 1.6, 2.0],
        help="cap factors for the Question 1 sweep (1.3 is the base case)",
    )
    ap.add_argument("--DP", action="store_true", help="run dynamic pricing and compare it with C")
    ap.add_argument("--sos1", action="store_true", help="DP: SOS1 instead of big-M")
    ap.add_argument("--reg", type=float, default=1e-3, help="DP: weight of price regularization towards C's duals")
    ap.add_argument("--time-limit", type=float, default=None)
    ap.add_argument("--bb", action="store_true", help="DP: impose budget balance")
    ap.add_argument("--ir", action="store_true", help="DP: impose soft individual rationality (also adds budget balance)")
    ap.add_argument("--ir-penalty", type=float, default=100.0, help="DP: penalty per DKK of IR slack")
    ap.add_argument("--threads", type=int, default=0, help="DP: Gurobi threads (0 = Gurobi default)")
    ap.add_argument("--tight-M", action="store_true", help="DP: use one M per complementarity set")
    ap.add_argument("--M-scale", type=float, default=1.0, help="DP: safety factor on the per-set M values")
    ap.add_argument(
        "--mip-focus",
        type=int,
        default=0,
        choices=(0, 1, 2, 3),
        help="DP: Gurobi MIPFocus",
    )
    ap.add_argument(
        "--ir-bench",
        choices=("R", "I", "best"),
        default="best",
        help="DP: individual-rationality benchmark for each member",
    )
    ap.add_argument(
        "--cap-f",
        type=float,
        nargs="+",
        default=[1.3],
        help="DP: cap tightness values (cap multiplied by f'/1.3; 1.3 is the base case)",
    )
    ap.add_argument("--q3", action="store_true", help="run the full Question 3 analysis (several hours)")
    ap.add_argument("--q3-step1", action="store_true", help="evaluate deterministic prices on out-of-sample scenarios")
    ap.add_argument("--q3-runs", action="store_true", help="solve the configured Question 3 experiments")
    ap.add_argument("--q3-evaluate", action="store_true", help="evaluate completed Question 3 runs out of sample")
    ap.add_argument("--q3-figures", action="store_true", help="draw Question 3 figures from completed runs")
    ap.add_argument("--q3-list", action="store_true", help="list Question 3 runs and completion status")
    ap.add_argument("--q3-only", nargs="*", help="restrict --q3-runs to these run names")
    ap.add_argument("--q3-time-scale", type=float, default=1.0, help="scale Question 3 solver time limits")
    ap.add_argument("--q3-threads", type=int, default=0, help="Gurobi threads per Question 3 run")
    ap.add_argument("--q3-n-out", type=int, default=100, help="out-of-sample scenarios for Question 3")
    ap.add_argument("--q3-seed-out", type=int, default=1000, help="out-of-sample seed for Question 3")
    ap.add_argument("--unit-tests", action="store_true", help="run the fast, solver-free test suite")
    ap.add_argument("--test", action="store_true", help="run Question 3 model and scenario validation checks")
    ap.add_argument("--test-time-limit", type=float, default=300.0, help="time limit per validation MIP")
    ap.add_argument("--test-only", nargs="*", help="run only these validation checks (V1-V6)")
    args = ap.parse_args()

    if args.unit_tests or args.test:
        run_unit_tests()
        if args.unit_tests and not args.test:
            return

    community = load_community(args.community, day=args.day)
    if args.final_soc == "initial":
        for m in community.members:
            if m.has_battery:
                m.battery_final_soc_kWh = m.battery_initial_soc_kWh
    print(community.summary(), "\n")
    if args.members:
        for member in community.members:
            print(member.summary(), "\n")
    print("Cap profile (kW):", " ".join(f"{value:.1f}" for value in community.contract.capacity_limit_kW))
    individual_caps = community.contract.individual_capacity_limit_kW
    print(
        "Individual caps in model I (kW), hour 0 / hour 11:",
        "  ".join(
            f"{consumer_id} ({community.member(consumer_id).archetype}) "
            f"{individual_caps[consumer_id][0]:.2f}/{individual_caps[consumer_id][11]:.2f}"
            for consumer_id in ("C1", "C1_PVB", "C4", "C4_PVB")
            if consumer_id in community.member_ids
        ),
    )
    (RESULTS_DIR / args.community).mkdir(parents=True, exist_ok=True)

    if args.demo:
        import runpy

        runpy.run_module("src.q3_scenario_generation", run_name="__main__")
    if args.DP:
        DynamicPricingExperiment(RESULTS_DIR).run(community, args)
    if args.R or args.I or args.C:
        mode = (
            ContractMode.COMMUNITY
            if args.C
            else ContractMode.INDIVIDUAL
            if args.I
            else ContractMode.NONE
        )
        result = CommunityModel(
            community,
            contract_mode=mode,
            battery_end_soc_value_DKK_per_kWh=args.battery_end_value,
        ).build().solve()
        print(result)
        result.save(RESULTS_DIR / args.community, tag=mode.value)
        plot_paths = plot_results(
            result,
            community.contract,
            community.energy_price,
            RESULTS_DIR / args.community,
        )
        print("Plots written:")
        for path in plot_paths:
            print(f"  {path}")
        if mode != ContractMode.NONE:
            reference = CommunityModel(
                community,
                contract_mode=ContractMode.NONE,
                battery_end_soc_value_DKK_per_kWh=args.battery_end_value,
            ).build().solve()
            benefit_path = plot_member_benefits(
                reference,
                result,
                community.members,
                RESULTS_DIR / args.community,
            )
            print(f"  {benefit_path}")
    if args.q1 or args.q1_base or args.q1_sweep or args.q1_settlement:
        Question1Experiment(RESULTS_DIR).run(community, args)
    if args.q3_list:
        from src.q3_analysis import RUNS, run_done

        runs_dir = RESULTS_DIR / args.community / "q3" / "runs"
        for name, cfg in RUNS.items():
            state = "solved" if run_done(runs_dir / name) else "not solved"
            print(f"{name:12s} {state:10s} {cfg['time_limit']:5d}s")
    if args.q3 or args.q3_step1 or args.q3_runs or args.q3_evaluate or args.q3_figures:
        from src import q3_analysis

        q3_community = load_community(args.community)
        q3_out = RESULTS_DIR / args.community / "q3"
        q3_out.mkdir(parents=True, exist_ok=True)
        if args.q3 or args.q3_step1:
            q3_analysis.step1(q3_community, q3_out, args.q3_n_out, args.q3_seed_out)
        if args.q3 or args.q3_runs:
            q3_analysis.run_all(q3_community, q3_out, args.q3_only, scale=args.q3_time_scale,
                                threads=args.q3_threads)
        if args.q3 or args.q3_evaluate:
            q3_analysis.evaluate_all(q3_community, q3_out, args.q3_n_out, args.q3_seed_out)
        if args.q3 or args.q3_figures:
            from src import q3_figures

            q3_figures.main(["--community", args.community])
    if args.test:
        from src import q3_validation

        validation_args = ["--community", args.community, "--time-limit", str(args.test_time_limit)]
        if args.test_only is not None:
            validation_args.extend(["--only", *args.test_only])
        q3_validation.main(validation_args)


if __name__ == "__main__":
    main()
