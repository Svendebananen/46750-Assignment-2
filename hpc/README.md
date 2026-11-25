# Question 3 on the DTU HPC

The runs of `src/q3_experiments.py` as LSF jobs: one job per run, each started when the run it starts from is
done, so independent chains run in parallel (DP_BB -> DP_BB_IR -> SDPx_N5 -> CVaR, N=3, 7, 10, and
eps=0 -> 0.2 -> 0.4). A last job evaluates every run on the 100 out-of-sample scenarios and draws figures 2-4.

## 1. Log in and get the code (once)

```bash
ssh <DTU-user>@login1.hpc.dtu.dk          # the login address in DTU's HPC guide
linuxsh                                   # an interactive node for the setup (not the login node)
git clone https://github.com/Svendebananen/46750-Assignment-2.git
cd 46750-Assignment-2
git checkout q3-stochastic-pricing
```

## 2. Python environment and licence check (once)

```bash
module avail gurobi       # pick a 13.x version and write it in hpc/config.sh (GUROBI_MODULE)
module avail python3      # write a 3.10+ version in hpc/config.sh (PYTHON_MODULE)
bash hpc/setup_q3.sh      # creates .venv-hpc and solves a 3000-variable test model
```

`licence OK` at the end means Gurobi found the cluster's licence. A licence error means the module does not
provide one: ask the HPC support, or use a free Gurobi academic WLS licence (`~/gurobi.lic`; at most 2 sessions
at a time, the compute nodes need internet access).

## 3. Results already solved on the PC (optional)

Runs that are already solved are skipped. To reuse the PC's runs (e.g. to solve only N = 3 and N = 7, which start
from SDPx_N5), copy them from the PC (the results folder is not in git):

```bash
scp -r results/base/q3/runs <DTU-user>@login1.hpc.dtu.dk:46750-Assignment-2/results/base/q3/
```

## 4. Submit

```bash
bash hpc/submit_q3.sh                     # every run that is not solved yet, then the evaluation
bash hpc/submit_q3.sh SDPx_N3 SDPx_N7     # only these
bjobs                                     # the queue: PEND = waiting (queue or dependency), RUN = running
```

Logs: `hpc/logs/q3_<run>_<jobid>.out` (and `.err`), Gurobi's log: `results/base/q3/runs/<run>/gurobi.log`.
Results: `results/base/q3/` (comparison table, figures). Copy them back to the PC with `scp -r` the other way.

## Settings (`hpc/config.sh`)

Cores per run (`CORES`, passed to Gurobi as threads), memory per core, wall time per run (`WALL`, 1:30 covers the
900 s start phase + 1800 s solve) and the queue. The time limits of the runs themselves are in `RUNS` of
`src/q3_experiments.py`.
