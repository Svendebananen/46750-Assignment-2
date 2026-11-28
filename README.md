# 46750 — Assignment 2

The project includes the community data, contract and dispatch models, dynamic pricing, and stochastic pricing experiments for all three assignment questions.

## Setup

Use Python 3.11 or later. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

Alternatively, create the Conda environment with `conda env create -f environment.yaml` and activate it with
`conda activate 46750-a2`.

Solving the full community models requires an active Gurobi academic or unrestricted license. Follow Gurobi's
[academic license instructions](https://www.gurobi.com/academia/academic-program-and-licenses/). The license bundled
with the package is size-limited and cannot solve the full community models.

## Run the questions

`main.py` is the command-line entry point. Use `python main.py --help` for all options.

| Question | Command | Purpose |
|---|---|---|
| 1 | `python main.py --q1` | Run the comparison, contract sweep, and settlement analyses |
| 2 | `python main.py --q2-dp --q2-ir` | Run Question 2 dynamic pricing with budget balance and individual-rationality requirements |
| 3 | `python main.py --q3` | Run the scenario study, stochastic experiments, out-of-sample evaluation, and figures |
| Tests | `python main.py --unit-tests` | Run fast data-loader and CLI tests without solving models |
| Question 3 validation | `python main.py --q3-validation` | Run the unit tests and Question 3 scenario/model checks |

Question 1 experiments can be run separately with `--q1-base`, `--q1-sweep`, or `--q1-settlement`. Question 2 options
use the `--q2-` prefix, for example `--q2-bb`, `--q2-ir`, `--q2-cap-f`, `--q2-sos1`, and `--q2-time-limit`.
For individual Question 1 dispatch cases, use `--q1-r`, `--q1-i`, or `--q1-c`. The former unprefixed options
(such as `--DP`, `--ir`, `--time-limit`, `--R`, and `--C`) remain accepted as aliases.

Question 3's full run solves several models and can take hours. Run its stages separately with `--q3-step1`,
`--q3-runs`, `--q3-evaluate`, or `--q3-figures`; use `--q3-list` to see run status. `--q3-time-scale` multiplies
per-run solver time limits; it is not a total runtime limit. Use it with `--q3` or `--q3-runs`, and use
`--q3-only RUN_NAME ...` to select runs. The full Question 3 validation takes approximately 15–20 minutes with the
default solver limits; use `--q3-validation --q3-validation-only V1 V2` or
`--q3-validation --q3-validation-time-limit 60` to shorten it. The old `--test`, `--test-only`, and
`--test-time-limit` validation options remain accepted as aliases.

Results and figures are written under `results/q1`, `results/q2`, and
`results/q3`, according to the question that produced them. The default community is `base`; select
another community file with `--community NAME`, or load a different day with `--day d1` through `--day d7` for the
deterministic models. The stochastic Question 3 study uses day `d1`.

## Project layout

```text
main.py                 Short executable entry point
src/cli.py              Command-line parsing and dispatch
src/data_loader.py      Community and input-data loading
src/model.py            Shared community optimization models
src/q1_*.py             Question 1 analysis and figures (shared dispatch plots included)
src/q2_*.py             Question 2 dynamic-pricing model, analysis, and figures
src/q3_*.py             Question 3 scenarios, stochastic analysis, validation, and figures
data/                   Input parameters, profiles, and assignment figures
hpc/                    DTU HPC setup and job-submission scripts
tests/                  Fast unit and CLI smoke tests
```

The shipped data describes one 16-member base community. Input files are in `data/`; experiments write generated
tables, model outputs, and plots to `results/`.

## DTU HPC
The High performance compute provided by DTU is convenient to use for the large models in question 2 and 3.
Question 2 and 3 have submission scripts in `hpc/`. 
For cluster setup, result transfer, and job submission, see the short [HPC guide](hpc/README.md). 

## Docstring of functions added

In the descriptions below, hourly energy
flows and demand are in kWh per one-hour interval (numerically equal to the
average kW over that hour); power and capacity limits are in kW; energy prices
and tariffs are in DKK/kWh; costs, revenues, and settlements are in DKK; and
battery terminal values are in DKK/kWh. Indices, probabilities, ratios, flags,
and solver settings are dimensionless. 

### `src/cli.py`

- `run_unit_tests()` — Runs the fast unit-test suite; returns no value.
- `main()` — Parses command-line options and dispatches the requested experiment; returns no value.

### `src/model.py`

- `Results.save(folder, tag)` — Inputs: output folder and filename tag. Writes hourly results and a summary; no return value.
- `Results.__str__()` — Input: solved results. Returns a text summary with objective (DKK) and daily energy totals (kWh).
- `_status_name(code)` — Input: Gurobi status code. Returns its status name as text.
- `SingleConsumerModel.__init__(data, name, verbose, battery_end_soc_value_DKK_per_kWh)` — Inputs: consumer data and solver settings. Initializes a single-consumer model; no return value.
- `SingleConsumerModel.build()` — Builds the single-consumer model; returns the model instance.
- `SingleConsumerModel._add_variables()` — Adds hourly import, export, load, PV, deviation, and optional battery variables; no return value.
- `SingleConsumerModel._objective()` — Builds the consumer cost objective (DKK); returns a Gurobi expression.
- `SingleConsumerModel._objective_sense()` — Returns the minimization direction; no physical units.
- `SingleConsumerModel._power_balance(t)` — Input: hour index. Returns the hourly energy-balance expression (kWh).
- `SingleConsumerModel._add_constraints()` — Adds load, PV, deviation, daily-energy, and optional battery constraints; no return value.
- `SingleConsumerModel._add_battery_constraints()` — Adds battery charge/discharge and state-of-charge constraints; no return value.
- `SingleConsumerModel.solve()` — Solves the model; returns `Results` with hourly values and objective (DKK).
- `SingleConsumerModel._extract_results(status)` — Inputs: solver status. Returns `Results` with hourly flows (kWh), prices (DKK/kWh), and objective (DKK).
- `CommunityModel.__new__(data, contract_mode, name, verbose, battery_end_soc_value_DKK_per_kWh)` — Inputs: community data and contract/solver settings. Returns the matching community model instance.
- `_CommunityModelBase.__init__(data, name, verbose, battery_end_soc_value_DKK_per_kWh)` — Inputs: community data and solver settings. Initializes shared model state; no return value.
- `_CommunityModelBase.build()` — Builds the community model and constraints; returns the model instance.
- `_CommunityModelBase._add_variables()` — Adds member load, deviation, PV, battery, import, and export variables; no return value.
- `_CommunityModelBase._add_contract_variables()` — Adds member import/export variables; no return value.
- `_CommunityModelBase._objective()` — Abstract objective hook; returns a Gurobi cost expression (DKK).
- `_CommunityModelBase._member_var(name, i, t)` — Inputs: variable name, member index, and hour index. Returns the requested model variable or zero.
- `_CommunityModelBase._add_physical_constraints()` — Adds member power-balance constraints (kWh) and member constraints; no return value.
- `_CommunityModelBase._add_member_constraints()` — Adds load, deviation, PV, and optional daily-energy/battery constraints; no return value.
- `_CommunityModelBase._add_contract_constraints()` — Contract-constraint hook; no return value.
- `_CommunityModelBase._add_member_battery_constraints(i, d)` — Inputs: member index and data. Adds battery limits (kW/kWh) and state-of-charge constraints; no return value.
- `_CommunityModelBase._battery_terminal_value()` — Returns the terminal battery-value contribution to the objective (DKK).
- `_CommunityModelBase._objective_sense()` — Returns the minimization direction; no physical units.
- `_CommunityModelBase._extract_results(status)` — Input: solver status. Returns community `Results`, with hourly flows (kWh), member costs (DKK), and objective (DKK).
- `_CommunityModelBase._member_operating_costs()` — Returns each member's operating cost (DKK), keyed by consumer ID.
- `_CommunityModelBase.solve()` — Solves the community model; returns `Results`.
- `NoContractCommunityModel._objective()` — Builds the no-contract community cost objective (DKK); returns a Gurobi expression.
- `IndividualContractCommunityModel._add_contract_variables()` — Adds variables for individual contracts; no return value.
- `IndividualContractCommunityModel._add_contract_constraints()` — Adds individual import-cap and penalty constraints; no return value.
- `IndividualContractCommunityModel._objective()` — Builds the individual-contract cost objective (DKK); returns a Gurobi expression.
- `CommunityContractCommunityModel._add_physical_constraints()` — Adds the community balance constraint; no return value.
- `CommunityContractCommunityModel._add_contract_variables()` — Adds community import/export, net-consumption, and excess variables; no return value.
- `CommunityContractCommunityModel._add_contract_constraints()` — Adds community capacity and excess constraints; no return value.
- `CommunityContractCommunityModel._extract_results(status)` — Input: solver status. Returns community-contract `Results` with hourly flows and costs.
- `CommunityContractCommunityModel._objective()` — Builds the community-contract cost objective (DKK); returns a Gurobi expression.
- `__getattr__(name)` — Input: missing attribute name. Forwards supported compatibility lookups; returns the resolved attribute.

### `src/q1_analysis.py`

- `with_contract(com, beta, cap_scale)` — Inputs: community, tariff discount ratio, and cap multiplier. Returns a copy with modified contract terms; ratios are dimensionless and caps are kW.
- `solve(com, mode, end_value)` — Inputs: community, contract mode, and battery terminal value (DKK/kWh). Returns the built model and its `Results`.
- `_arr(var, T)` — Inputs: model variables and hour indices. Returns their solved values as a numeric array.
- `asset_label(m)` — Input: member data. Returns its PV/battery asset label.
- `dso_metrics(imp, imp_ref, cap)` — Inputs: import and cap profiles (kW). Returns peak/total import (kW/kWh), reduction (%), and cap-exceedance metrics (kW/kWh/hours).
- `dso_revenue(mdl, com, mode)` — Inputs: solved model, community, and contract mode. Returns DSO grid revenue (DKK).
- `base_comparison(com, end_value)` — Inputs: community and battery terminal value (DKK/kWh). Returns the base comparison for contract cases R, I, and C.
- `contract_sweep(com, end_value, betas, phis)` — Inputs: community, terminal value (DKK/kWh), discount ratios, and cap factors. Returns a `DataFrame` of contract gains and DSO metrics.
- `own_costs(mdl, end_value)` — Inputs: solved model and terminal value (DKK/kWh). Returns member operating costs excluding grid payments (DKK).
- `grid_payment_RI(mdl, com, mode)` — Inputs: solved model, community, and R/I mode. Returns each member's grid payment (DKK).
- `community_settlement(mdl, com, end_value)` — Inputs: solved C model, community, and terminal value (DKK/kWh). Returns member settlement and manager-surplus results (DKK).
- `settlement_study(com, end_value)` — Inputs: community and terminal value (DKK/kWh). Returns the R/I/C individual-rationality and budget study.
- `Question1Experiment.__init__(results_dir)` — Input: results directory. Initializes Question 1 output paths; no return value.
- `Question1Experiment.run(community, args)` — Inputs: community and CLI options. Runs selected Question 1 experiments; no return value.
- `Question1Experiment._run_base(community, end_value, output_dir)` — Inputs: community, terminal value (DKK/kWh), and output directory. Runs the base comparison; no return value.
- `Question1Experiment._run_sweep(community, end_value, output_dir, plotter, betas, phis)` — Inputs: community, sweep parameters, output path, and plotter. Runs the contract sweep; no return value.
- `Question1Experiment._run_settlement(community, end_value, output_dir, plotter)` — Inputs: community, terminal value, output path, and plotter. Runs the settlement study; no return value.

### `src/q1_plotting.py`

- `plot_results(result, contract, energy_price, output_dir, tag)` — Inputs: model results, contract, and hourly energy prices (DKK/kWh). Saves dispatch/capacity figures and returns their paths.
- `plot_member_benefits(reference, contract_result, members, output_dir)` — Inputs: reference and contract results and member data. Saves a benefit figure (DKK) and returns its path.
- `_zero_crossing(x, y)` — Inputs: numeric curve coordinates. Returns the first interpolated zero crossing, or `None`.
- `plot_gain_map(grid, output_dir, base)` — Inputs: sweep data and base contract ratios. Saves a community-gain figure and returns its path.
- `plot_dso_view(grid, output_dir, beta, base_phi)` — Inputs: sweep data and contract ratios. Saves DSO revenue/exceedance figures and returns its path.
- `plot_member_gains(members, output_dir)` — Inputs: member benefit data (DKK) and output directory. Saves a figure and returns its path.
- `Question1Plotter.__init__(output_dir)` — Input: output directory. Initializes the plotting helper; no return value.
- `Question1Plotter.gain_map(grid)` — Input: gain-sweep data. Saves the gain map and returns its path.
- `Question1Plotter.dso_view(grid)` — Input: DSO sweep data. Saves the DSO view and returns its path.
- `Question1Plotter.member_gains(members)` — Input: member benefit data. Saves the member-gain figure and returns its path.

### `src/q2_analysis.py`

- `solve_dp(com, args, lam_C, label, budget_balance, ir_benchmark)` — Inputs: community, CLI/model settings, C prices (DKK/kWh), label, and optional requirements. Builds/solves a DP variant; returns `None` if no solution is available.
- `asset_type(m)` — Input: member data. Returns the PV/battery asset label.
- `member_order(members)` — Input: member records. Returns indices ordered by asset type and load slope.
- `ir_benchmark_costs(R, I, rule)` — Inputs: R/I results and benchmark rule. Returns member benchmark costs (DKK).
- `q2_metrics(r, com, prices)` — Inputs: run results, community, and prices (DKK/kWh). Returns community metrics (DKK/day, kWh/day, and kW).
- `member_table(r, R, I, bench, members)` — Inputs: run, R/I references, benchmark costs (DKK), and members. Returns per-member cost and benefit table (DKK).
- `validation_row(r, lam_C)` — Inputs: DP results and C prices (DKK/kWh). Returns model/contract validation metrics.
- `write_latex(df, path, caption, label, digits)` — Inputs: table and LaTeX metadata. Writes a LaTeX table; returns no value.
- `write_latex.fmt(v)` — Input: a table cell. Returns its formatted LaTeX representation.
- `scaled_cap(com, f)` — Inputs: community and dimensionless cap factor. Returns a copy with scaled community/member capacities (kW).
- `dual_settlement_costs(C, lam_C)` — Inputs: C results and internal prices (DKK/kWh). Returns member settlement costs (DKK).
- `run_dp_case(com, args, out, plots)` — Inputs: community, run settings, output path, and plot flag. Returns results for C/R/I and DP steps.
- `run_dp(com, args, results_dir)` — Inputs: community, run settings, and results directory. Runs Question 2 experiments; no return value.
- `DynamicPricingExperiment.__init__(results_dir)` — Input: results directory. Initializes Question 2 output paths; no return value.
- `DynamicPricingExperiment.run(community, args)` — Inputs: community and CLI options. Runs selected Question 2 experiments; no return value.

### `src/q2_model.py`

- `_battery_parameters(member)` — Input: member data. Returns efficiencies (dimensionless), charge/discharge limits (kW), capacity (kWh), and initial state of charge (kWh).
- `DynamicPricingCommunityModel.__init__(data, name, verbose, battery_end_soc_value_DKK_per_kWh, method, M_P, M_D, reg_weight, reg_reference, mip_gap, time_limit, budget_balance, ir_benchmark, ir_penalty, tight_M, M_scale, mip_focus, threads)` — Inputs: community/model settings, solver limits, and optional budget/IR requirements. Initializes a DP model; no return value.
- `DynamicPricingCommunityModel.build()` — Builds model C, posted prices, KKT conditions, and optional requirements; returns the model.
- `DynamicPricingCommunityModel._add_price()` — Adds hourly posted-price variables (DKK/kWh); no return value.
- `DynamicPricingCommunityModel._compl(expr, mu, name, M_P, M_D)` — Inputs: complementarity slack and multiplier bounds. Adds SOS1 or big-M complementarity constraints; no return value.
- `DynamicPricingCommunityModel._add_kkt()` — Adds member KKT stationarity and complementarity constraints; no return value.
- `DynamicPricingCommunityModel.dual_objective(i)` — Input: member index. Returns the member's dual-cost expression (DKK).
- `DynamicPricingCommunityModel._add_budget_balance()` — Adds the budget-balance constraint; no return value.
- `DynamicPricingCommunityModel._add_regularization()` — Adds price regularization toward a reference price (DKK/kWh); no return value.
- `DynamicPricingCommunityModel._member_operating_costs()` — Returns each member's DP operating cost (DKK).
- `DynamicPricingCommunityModel._extract_results(status)` — Input: solver status. Returns DP `Results` with posted prices (DKK/kWh), validation, and cost/surplus summaries (DKK).
- `DynamicPricingCommunityModel.solve()` — Solves the DP model; returns `Results`.
- `DynamicPricingCommunityModel.validate()` — Returns complementarity, big-M, price-bound, and indifferent-pair checks; residuals use model units.
- `DynamicPricingCommunityModel._add_individual_rationality()` — Adds soft individual-rationality constraints and compensation slack (DKK); no return value.
- `internal_price_of_C(c_model)` — Input: solved community-contract model. Returns hourly internal prices (DKK/kWh).

### `src/q2_plotting.py`

- `_cap_binding_hours(hourly, cap, tol)` — Inputs: hourly imports (kWh), cap (kW), and tolerance. Returns a Boolean mask of cap-binding hours.
- `_meter_prices(community)` — Input: community data. Returns hourly marginal import/export prices at the meter (DKK/kWh).
- `plot_dp_prices(result, lambda_C, community, output_dir, tag)` — Inputs: DP results, C prices, and community. Saves price/flow figures and returns the path.
- `plot_dp_compare_prices(runs, lambda_C, community, output_dir, tag)` — Inputs: multiple run results, C prices, and community. Saves comparison figures and returns the path.
- `_tex(name)` — Input: model label. Returns its LaTeX-formatted text.
- `_min_span(ax, span)` — Inputs: plot axis and minimum span. Adjusts plot range; returns no value.
- `plot_dp_cost_of_requirements(table, output_dir, tag)` — Inputs: requirement-cost data (DKK) and output options. Saves a figure and returns the path.
- `plot_dp_benefits_by_member(tables, benchmark, output_dir, tag)` — Inputs: member tables (DKK), benchmark, and output options. Saves a figure and returns the path.
- `plot_dp_cap_sweep(sweep, output_dir, tag)` — Inputs: cap-sweep results and output options. Saves a sensitivity figure and returns the path.
- `plot_dp_shift_heatmap(shifts, assets, cap, imports, output_dir, base, compare, tag)` — Inputs: member/hour load shifts (kWh), assets, cap (kW), and comparison runs. Saves a heatmap and returns the path.
- `DynamicPricingPlotter.__init__(output_dir)` — Input: output directory. Initializes the plotting helper; no return value.
- `DynamicPricingPlotter.prices(result, lambda_C, community, tag)` — Inputs: run result, reference prices, and community. Saves price plots and returns the path.
- `DynamicPricingPlotter.compare_prices(runs, lambda_C, community, tag)` — Inputs: run results, reference prices, and community. Saves comparison plots and returns the path.
- `DynamicPricingPlotter.cost_of_requirements(table, tag)` — Input: requirement-cost table (DKK). Saves its figure and returns the path.
- `DynamicPricingPlotter.benefits_by_member(tables, benchmark, tag)` — Inputs: member tables (DKK) and benchmark. Saves its figure and returns the path.
- `DynamicPricingPlotter.cap_sweep(sweep, tag)` — Input: cap-sweep results. Saves its figure and returns the path.
- `DynamicPricingPlotter.shift_heatmap(shifts, assets, cap, imports, base, compare, tag)` — Inputs: load shifts (kWh), assets, cap (kW), and comparison runs. Saves its figure and returns the path.

### `src/q3_analysis.py`

- `deterministic_dp(com, time_limit)` — Inputs: community and optional solver limit (s). Solves DP on expected parameters; returns the solved model.
- `c_prices(com)` — Input: community. Returns C's hourly internal prices (DKK/kWh).
- `step1(com, out, n_out, seed_out)` — Inputs: community, output path, scenario count, and random seed. Runs the out-of-sample setup and returns a metrics `DataFrame`.
- `in_sample(com, n, seed)` — Inputs: community, scenario count, and seed. Returns generated in-sample scenarios.
- `run_done(folder)` — Input: run directory. Returns whether the run has completed.
- `load_prices(folder)` — Input: run directory. Returns saved hourly prices (DKK/kWh).
- `solve_run(com, name, cfg, runs_dir, lam, scale, threads)` — Inputs: community, run configuration, reference prices (DKK/kWh), solver scale, and thread count. Solves/saves one run; no return value.
- `run_all(com, out, only, force, scale, threads)` — Inputs: community, output directory, run selection, and solver options. Executes selected runs; no return value.
- `evaluate_all(com, out, n_out, seed_out)` — Inputs: community, output directory, scenario count, and seed. Evaluates runs out of sample and returns a metrics `DataFrame`.
- `write_comparison_tex(table, path)` — Inputs: comparison table and output path. Writes a LaTeX table; no return value.
- `main()` — Parses Question 3 analysis options and dispatches stages; returns no value.

### `src/q3_plotting.py`

- `load_run(folder)` — Input: run directory. Returns evaluated run data, or `None` if unavailable.
- `load_runs(runs_dir)` — Input: runs directory. Returns available run data keyed by run name.
- `_mean(df, col)` — Inputs: results table and column name. Returns the column mean in the same units as that column.
- `fig_models(runs, out, tag)` — Inputs: run data, output directory, and tag. Saves the model-comparison figure and returns its path (or `None`).
- `fig_epsilon(runs, out, tag)` — Inputs: run data, output directory, and tag. Saves the epsilon comparison figure and returns its path (or `None`).
- `fig_scenarios_N(runs, out, tag)` — Inputs: run data, output directory, and tag. Saves the scenario-count figure and returns its path (or `None`).
- `plot_q3_validation(scenarios, base_members, cap, in_model_cost, evaluator_cost, dp_cost, output_dir, tag)` — Inputs: scenario/member data, cap (kW), and costs (DKK). Saves the validation plot and returns its path.
- `plot_q3_validation.net_demand(members)` — Input: member flow data (kWh). Returns net demand by hour.
- `plot_q3_uncertainty_impact(ev, predicted_cost, predicted_surplus, output_dir, label, tag)` — Inputs: evaluation results and predicted cost/surplus (DKK). Saves the uncertainty-impact plot and returns its path.
- `main(argv)` — Inputs: optional command-line arguments. Runs Question 3 plotting; returns no value.

### `src/q3_scenario_data.py`

- `Scenario.member(consumer_id)` — Input: consumer ID. Returns that member's scenario data.
- `apply_theta(member, theta_profile, theta_cL)` — Inputs: member data and dimensionless uncertainty multipliers. Returns scaled member data.
- `scenario_members(com, theta)` — Inputs: community and per-member uncertainty multipliers. Returns scenario member data.
- `scenario_community(com, scenario)` — Inputs: community and scenario. Returns a community copy with scenario members.
- `make_spec(com, rules)` — Inputs: community and optional uncertainty rules. Returns the per-member scenario-generation specification.
- `draw_scenarios(com, n, seed, quantities, spec)` — Inputs: community, scenario count, seed, uncertain quantities, and optional specification. Returns equiprobable scenarios.
- `expected_scenario(com)` — Input: community. Returns the deterministic baseline scenario (all multipliers equal to one).
- `check_scenarios(com, scenarios, spec, tol)` — Inputs: community, scenarios, specification, and tolerance. Checks scenario invariants; returns no value or raises `AssertionError`.
- `save_scenarios(scenarios, path, seed)` — Inputs: scenarios, output path, and optional seed. Writes scenario parameters/probabilities and returns the path.

### `src/q3_stochastic.py`

- `_ScenarioBlock.__init__(data, m, price, s, **dp_kwargs)` — Inputs: scenario community, shared solver model and prices, scenario index, and DP options. Initializes one scenario block; no return value.
- `_ScenarioBlock.build()` — Adds one scenario's physical model/KKT constraints; returns the block.
- `_ScenarioBlock.community_cost()` — Returns the scenario community-cost expression (DKK).
- `_ScenarioBlock.meter_bill()` — Returns the scenario meter-bill expression (DKK).
- `_ScenarioBlock.payments()` — Returns members' total payments to the manager (DKK).
- `StochasticResult.summary()` — Input: stochastic solve results. Returns a text summary with costs/surplus (DKK), exceedance (kWh), solver gap, and runtime (s).
- `_status(m)` — Input: Gurobi model. Returns its solver status as text.
- `StochasticDynamicPricingModel.__init__(com, scenarios, name, objective, cvar_alpha, cvar_weight, constraints, epsilon, budget_balance, ir_benchmark, ir_penalty, reg_weight, reg_reference, method, tight_M, M_scale, M_P, M_D, battery_end_soc_value_DKK_per_kWh, mip_gap, time_limit, mip_focus, bb_tol, start_prices, start_box, start_time_limit, start_solution, threads, verbose)` — Inputs: community, scenarios/probabilities, risk and requirement settings, price references (DKK/kWh), solver options, and start solutions. Initializes the model; no return value.
- `StochasticDynamicPricingModel.build()` — Builds the shared price and scenario models plus risk/requirement constraints; returns the model.
- `StochasticDynamicPricingModel._chance()` — Returns whether chance constraints are active; Boolean output.
- `StochasticDynamicPricingModel._add_requirements()` — Adds budget-balance and individual-rationality constraints; no return value.
- `StochasticDynamicPricingModel._objective()` — Builds the expected-cost or CVaR objective plus penalties (DKK); returns a Gurobi expression.
- `StochasticDynamicPricingModel.size()` — Returns counts of scenarios, members, variables, constraints, and complementarity pairs.
- `StochasticDynamicPricingModel.warm_start(prices, time_limit, box)` — Inputs: hourly prices (DKK/kWh), time limit (s), and price range (DKK/kWh). Attempts a feasible solver start; returns success as a Boolean.
- `StochasticDynamicPricingModel.core_solution()` — Returns the solved core-variable values as a numeric array.
- `StochasticDynamicPricingModel.set_core_start(values, time_limit)` — Inputs: core-variable values and time limit (s). Attempts to complete a solver start; returns success as a Boolean.
- `StochasticDynamicPricingModel.solve()` — Solves the stochastic model; returns `StochasticResult` with price, cost, surplus, and scenario/member tables.
- `StochasticDynamicPricingModel._extract()` — Extracts the solved objective/prices and per-scenario/member results; returns `StochasticResult`.
- `benchmark_costs(com, scenarios, rule, battery_end_soc_value_DKK_per_kWh)` — Inputs: community, scenarios, benchmark rule, and terminal value (DKK/kWh). Returns each member's benchmark cost (DKK) per scenario.
- `_member_cost(model, i, x)` — Inputs: solved community model, member index, and primal solution. Returns that member's cost expression (DKK).
- `respond(com_s, prices, battery_end_soc_value_DKK_per_kWh, tie_break, tol)` — Inputs: scenario community, prices (DKK/kWh), terminal value (DKK/kWh), tie-break rule, and tolerance. Returns the solved member-response model.
- `evaluate_prices(com, prices, scenarios, benchmark, battery_end_soc_value_DKK_per_kWh, tie_break, label)` — Inputs: community, fixed prices (DKK/kWh), scenarios, optional benchmark costs (DKK), and evaluation options. Returns scenario-level and member-level `DataFrame`s.
- `risk_metrics(scen, members, alpha, tol)` — Inputs: scenario/member tables, tail share, and tolerance. Returns mean, quantile, CVaR, and risk/requirement metrics (cost/surplus in DKK).

### `src/q3_validation.py`

- `check(name, test, expected, measured, tolerance, ok, detail)` — Inputs: check labels/values and pass status. Prints and returns the validation row.
- `write_table(rows, folder)` — Inputs: validation rows and output directory. Writes CSV/LaTeX tables and returns both paths.
- `v1_scenarios(com, n, seed)` — Inputs: community, scenario count, and seed. Runs V1 scenario checks; returns pass status.
- `main(argv)` — Inputs: optional command-line arguments. Runs Question 3 validation; returns no value.

### `tests/test_cli.py`

- `EntryPointTests.run_cli(*args)` — Inputs: CLI arguments. Runs the entry point and returns its process result.
- `EntryPointTests.test_help_lists_all_question_and_test_entry_points()` — Checks that help lists the supported entry points; returns no value.
- `EntryPointTests.test_question_three_run_list_does_not_require_a_solve()` — Checks that listing Question 3 runs does not solve a model; returns no value.

### `tests/test_data_loader.py`

- `CommunityLoaderTests.test_base_community_has_consistent_hourly_data()` — Checks consistency of base-community hourly data; returns no value.
- `CommunityLoaderTests.test_loading_another_day_updates_profiles_consistently()` — Checks that selecting another day updates profiles consistently; returns no value.
- `CommunityLoaderTests.test_unknown_community_reports_a_clear_error()` — Checks that an unknown community produces a clear error; returns no value.
