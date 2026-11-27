# 46750 — Assignment 2

Optimization experiments for an energy community acting as a flexibility provider. The project includes the community
data, contract and dispatch models, dynamic pricing, and stochastic pricing experiments for all three assignment
questions.

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
| 2 | `python main.py --DP --ir` | Run dynamic pricing with budget balance and individual-rationality requirements |
| 3 | `python main.py --q3` | Run the scenario study, stochastic experiments, out-of-sample evaluation, and figures |
| Tests | `python main.py --unit-tests` | Run fast data-loader and CLI tests without solving models |
| Validation | `python main.py --test` | Run the unit tests and Question 3 scenario/model checks |

Question 1 experiments can be run separately with `--q1-base`, `--q1-sweep`, or `--q1-settlement`. Question 2 options
include `--bb`, `--ir`, `--cap-f`, `--sos1`, and `--time-limit`. For an individual dispatch model, use `--R`, `--I`, or
`--C`.

Question 3's full run solves several models and can take hours. Run its stages separately with `--q3-step1`,
`--q3-runs`, `--q3-evaluate`, or `--q3-figures`; use `--q3-list` to see run status. `--q3-time-scale` scales solver
time limits, and `--q3-only RUN_NAME ...` selects runs. The full validation takes approximately 15–20 minutes with
the default solver limits; use `--test --test-only V1 V2` or `--test --test-time-limit 60` for a smaller check.

Results and figures are written under `results/<community>/`. The default community is `base`; select another
community file with `--community NAME`, or load a different day with `--day d1` through `--day d7` for the
deterministic models. The stochastic Question 3 study uses day `d1`.

## Project layout

```text
main.py                 Short executable entry point
src/cli.py              Command-line parsing and dispatch
src/data_loader.py      Community and input-data loading
src/model.py            Shared community optimization models
src/plotting.py         Shared and Question 3 figures
src/q1_*.py             Question 1 analysis and figures
src/q2_*.py             Question 2 dynamic-pricing model, analysis, and figures
src/q3_*.py             Question 3 scenarios, stochastic analysis, validation, and figures
data/                   Input parameters, profiles, and assignment figures
hpc/                    DTU HPC setup and job-submission scripts
tests/                  Fast unit and CLI smoke tests
```

The shipped data describes one 16-member base community. Input files are in `data/`; experiments write generated
tables, model outputs, and plots to `results/`.

## DTU HPC

For cluster setup, result transfer, and job submission, see the short [HPC guide](hpc/README.md). Question 2 and 3
have submission scripts in `hpc/`.
