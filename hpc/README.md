# Question 3 on the DTU HPC

The runs of `src/q3_experiments.py` as LSF jobs on the DTU HPC: one job per run, each started when the run it
starts from is done, so independent runs solve in parallel and the PC can be switched off. A last job evaluates
every run on the 100 out-of-sample scenarios and draws figures 2-4.

    DP_BB -> DP_BB_IR -> SDPx_N5 -> SDPcvar_N5, SDPx_N3, SDPx_N7, SDPx_N10      (each "->" = starts from)
    SDPeps0_N5 -> SDPeps20_N5 -> SDPeps40_N5

Every command below says where it runs:

| Prompt looks like | Where | Used for |
|---|---|---|
| `you@your-pc:...$` | a terminal **on the PC**, in the repository folder | `scp` to and from the cluster |
| `hpclogin1(user) $` | the cluster's **login node** | `git`, submitting jobs, `bjobs` |
| `n-62-...(user) $` | an **interactive node** (`linuxsh`) | the one-time setup |

`scp` must always run **on the PC** (it copies between the PC and the cluster); run on the cluster it finds nothing.

## 1. Log in (cluster)

```bash
ssh <DTU-user>@login1.hpc.dtu.dk
```

From outside DTU without the DTU VPN, the login needs an ssh key (+ its passphrase + the DTU password); on the DTU
network or the VPN the DTU password is enough. See https://www.hpc.dtu.dk/?page_id=4317.

## 2. One-time setup (cluster)

```bash
git clone https://github.com/Svendebananen/46750-Assignment-2.git
cd 46750-Assignment-2
git checkout q3-stochastic-pricing
linuxsh                                   # interactive node for the setup (exit returns to the login node)
cd ~/46750-Assignment-2
bash hpc/setup_q3.sh
```

`setup_q3.sh` loads the modules of `hpc/config.sh` (`gurobi/13.0.0`, `python3/3.12.11`), creates the virtual
environment `.venv-hpc` with `gurobipy==13.0.0` and solves a 3000-variable test model. It must end with

    Gurobi (13, 0, 0) licence OK

The cluster's Gurobi module provides the licence; nothing else is needed. If a module is missing
(`ERROR: Unable to locate a modulefile`), list the versions with `module avail gurobi` / `module avail python3` and
change `hpc/config.sh`.

Later updates of the code: `git pull` on the cluster. If git refuses because `hpc/config.sh` was edited by hand,
discard the edit first: `git checkout hpc/config.sh && git pull`.

## 3. Copy the runs already solved on the PC (PC)

The results are not in git. Runs that already exist on the cluster are skipped, and a run can only start when the
run it starts from is there (e.g. `SDPx_N3` needs `SDPx_N5`). To reuse the PC's runs, copy them **from the PC**:

```bash
ssh <DTU-user>@login1.hpc.dtu.dk "mkdir -p 46750-Assignment-2/results/base/q3"
scp -r results/base/q3/runs <DTU-user>@login1.hpc.dtu.dk:46750-Assignment-2/results/base/q3/
```

Check on the cluster: `ls ~/46750-Assignment-2/results/base/q3/runs/SDPx_N5` lists `meta.json`, `prices.csv`,
`core_solution.npy`, ... A run that was still solving on the PC is copied without `meta.json` and counts as not
solved.

## 4. Submit (cluster)

```bash
cd ~/46750-Assignment-2
bash hpc/submit_q3.sh                     # every run of RUNS that is not solved yet, then the evaluation
bash hpc/submit_q3.sh SDPx_N3 SDPx_N7     # only these runs (their start runs must be solved or submitted too)
```

Each run prints `already solved, skipped` or `submitted (after <run>)`, and the evaluation job comes last.

| Command | |
|---|---|
| `bjobs` | your jobs: `PEND` = waiting for the queue or for the run it starts from, `RUN` = solving |
| `bkill <jobid> ...` | cancel jobs (e.g. jobs submitted before the start runs were copied) |
| `tail -f results/base/q3/runs/<run>/gurobi.log` | follow a run's Gurobi log (`Ctrl+C` stops following) |
| `hpc/logs/q3_<run>_<jobid>.out` / `.err` | the job's output and errors |

The jobs keep running after you log out. `bjobs` says `No unfinished job found` when everything is done.

## Question 2: dynamic pricing

Question 2 can use the same HPC environment. After following sections 1-2 above (including `bash hpc/setup_q3.sh`
on an interactive node), submit the three `--DP --ir` variants as one LSF job:

```bash
cd ~/46750-Assignment-2
bash hpc/submit_q2.sh
```

The submitted command is `python main.py --DP --ir` with `--time-limit` and `--threads` set from `hpc/config.sh`.
`--ir` runs cap-only DP, budget-balanced DP, and budget-balanced DP with individual rationality; the configured
time limit applies separately to each DP solve. The default is 3600 seconds per solve (up to three hours total)
and a 3:30 job wall time. Adjust `Q2_TIME_LIMIT` or `Q2_WALL` there to suit the queue's limits. Logs are written to
`hpc/logs/q2_dp_ir_<jobid>.out` and `.err`; result tables, plots, and model outputs go to `results/base/`.

The HPC may improve the result by allowing longer solves and using the allocated cores, but a smaller reported
MIP gap or a better objective is not guaranteed. Compare the final `MIP gap` and runtime in the output or
`results/base/dynamic_q2_steps.csv` with the PC run. If a solve reaches its time limit, Gurobi will use its best
available incumbent when one exists.

To copy the Q2 outputs back to the PC, run this **on the PC** from the repository folder:

```bash
scp -r <DTU-user>@login1.hpc.dtu.dk:46750-Assignment-2/results/base results/
```

## 5. Bring the results back (PC)

Copy the new runs **from the PC**, then evaluate every run together and draw the figures there (about 3 min):

```bash
scp -r <DTU-user>@login1.hpc.dtu.dk:46750-Assignment-2/results/base/q3/runs/SDPx_N3 \
       <DTU-user>@login1.hpc.dtu.dk:46750-Assignment-2/results/base/q3/runs/SDPx_N7 results/base/q3/runs/
.venv/bin/python -m src.q3_experiments --evaluate
.venv/bin/python -m src.q3_figures
```

Evaluating on the PC keeps all runs in one table and one set of figures even when some were solved on the PC
and some on the cluster. (The cluster's evaluation job does the same with the runs that are on the cluster.)

## Settings (`hpc/config.sh`)

| Setting | Value | |
|---|---|---|
| `GUROBI_MODULE`, `PYTHON_MODULE` | `gurobi/13.0.0`, `python3/3.12.11` | same Gurobi major version as on the PC (13.0.3) |
| `CORES` | 8 | cores per run, passed to Gurobi as `--threads` |
| `MEM_PER_CORE` | 4GB | LSF memory is per core |
| `WALL` | 01:30 | wall time per job: 900 s start phase + 1800 s solve + margin |
| `QUEUE` | hpc | |

The time limits of the runs themselves are in `RUNS` of `src/q3_experiments.py`; raise `WALL` with them.
