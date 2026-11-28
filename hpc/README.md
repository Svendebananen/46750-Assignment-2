# Questions 2 and 3 on the DTU HPC

This guide covers running Question 2 dynamic pricing and Question 3 stochastic pricing on the DTU HPC. Question 2
submits its three dynamic-pricing variants as one LSF job. Question 3 submits one job per run, with dependent runs
waiting for their starting run; a final job evaluates all runs on the 100 out-of-sample scenarios and draws figures
2-4. Independent Question 3 runs solve in parallel, so you can switch off your PC after submitting.

    DP_BB -> DP_BB_IR -> SDPx_N5 -> SDPcvar_N5, SDPx_N3, SDPx_N7, SDPx_N10      (each "->" = starts from)
    SDPeps0_N5 -> SDPeps20_N5 -> SDPeps40_N5

## Where to run commands

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

## 2. Set up the shared HPC environment (cluster)

Clone the repository on the cluster and check out `main`. Then start an interactive node and create the shared
environment used by both Question 2 and Question 3:

```bash
git clone https://github.com/Svendebananen/46750-Assignment-2.git
cd 46750-Assignment-2
git checkout main
linuxsh                                   # interactive node for the setup (exit returns to the login node)
cd ~/46750-Assignment-2
bash hpc/setup.sh
```

`setup.sh` loads the modules of `hpc/config.sh` (`gurobi/13.0.0`, `python3/3.12.11`), creates the virtual
environment `.venv-hpc` with `gurobipy==13.0.0` and solves a 3000-variable test model. It must end with

    Gurobi (13, 0, 0) licence OK

The cluster's Gurobi module provides the licence; nothing else is needed. If a module is missing
(`ERROR: Unable to locate a modulefile`), list the versions with `module avail gurobi` / `module avail python3` and
change `hpc/config.sh`.

For later code updates, run `git pull` on the cluster while on `main`. If git refuses because `hpc/config.sh` was
edited by hand, discard that edit first:

```bash
git checkout hpc/config.sh && git pull
```

## 3. Submit Question 2 and Question 3 jobs (cluster)

Question 2 needs no input files copied to the cluster. For Question 3, you can optionally upload runs already solved
on the PC before submitting. These are two different transfers: this optional copy sends PC results **to the
cluster**; step 4 copies cluster results back to the PC.

Question 3 results are not in git. Runs that already exist on the cluster are skipped, and a run can only start when
the run it starts from is there (e.g. `SDPx_N3` needs `SDPx_N5`). To reuse runs already solved on the PC, run this
**on the PC**:

```bash
ssh <DTU-user>@login1.hpc.dtu.dk "mkdir -p 46750-Assignment-2/results/base/q3"
scp -r results/base/q3/runs <DTU-user>@login1.hpc.dtu.dk:46750-Assignment-2/results/base/q3/
```

Check on the cluster: `ls ~/46750-Assignment-2/results/base/q3/runs/SDPx_N5` lists `meta.json`, `prices.csv`,
`core_solution.npy`, ... A run that was still solving on the PC is copied without `meta.json` and counts as not
solved.

Run either or both submission commands from the repository folder on the cluster:

```bash
cd ~/46750-Assignment-2
bash hpc/submit_q2.sh                     # Question 2: the three dynamic-pricing variants in one LSF job
bash hpc/submit_q3.sh                     # Question 3: every unsolved run, then the evaluation
bash hpc/submit_q3.sh SDPx_N3 SDPx_N7     # Question 3: only these runs (their start runs must be solved or submitted too)
```

Question 2 submits `python main.py --q2-dp --q2-ir` with `--q2-time-limit` and `--q2-threads` set from
`hpc/config.sh`. `--q2-ir` runs cap-only DP, budget-balanced DP, and budget-balanced DP with individual rationality;
the configured time limit applies separately to each DP solve. The default is 3600 seconds per solve (up to three
hours total) and a 3:30 job wall time. Adjust `Q2_TIME_LIMIT` or `Q2_WALL` there to suit the queue's limits. Logs are
written to `hpc/logs/q2_dp_ir_<jobid>.out` and `.err`; result tables, plots, and model outputs go to `results/base/`.

The HPC may improve the Question 2 result by allowing longer solves and using the allocated cores, but a smaller
reported MIP gap or a better objective is not guaranteed. Compare the final `MIP gap` and runtime in the output or
`results/base/dynamic_q2_steps.csv` with the PC run. If a solve reaches its time limit, Gurobi will use its best
available incumbent when one exists.

For Question 3, each run prints `already solved, skipped` or `submitted (after <run>)`, and the evaluation job comes
last. The evaluation job runs after all submitted runs, evaluates every run out of sample, and draws figures 2-4.

| Command or file | Description |
|---|---|
| `bjobs` | Your jobs: `PEND` = waiting for the queue or for the run it starts from, `RUN` = solving |
| `bkill <jobid> ...` | Cancel jobs (e.g. jobs submitted before the start runs were copied) |
| `tail -f results/base/q3/runs/<run>/gurobi.log` | Follow a run's Gurobi log (`Ctrl+C` stops following) |
| `hpc/logs/q3_<run>_<jobid>.out` / `.err` | Question 3 job output and errors |
| `hpc/logs/q2_dp_ir_<jobid>.out` / `.err` | Question 2 job output and errors |

The jobs keep running after you log out. `bjobs` says `No unfinished job found` when everything is done.

## 4. Download / copy results to the PC

Run these copy commands **on the PC**, from the repository folder.

### Question 2 outputs

Copy the results directory from the cluster:

```bash
scp -r <DTU-user>@login1.hpc.dtu.dk:46750-Assignment-2/results/base results/
```

This includes Question 2 result tables, plots, and model outputs in `results/base/`.

### Question 3 runs and evaluation

Copy the new runs from the cluster, then evaluate every run together and draw the figures on the PC (about 3 min):

```bash
scp -r <DTU-user>@login1.hpc.dtu.dk:46750-Assignment-2/results/base/q3/runs/SDPx_N3 \
       <DTU-user>@login1.hpc.dtu.dk:46750-Assignment-2/results/base/q3/runs/SDPx_N7 results/base/q3/runs/
.venv/bin/python -m src.q3_analysis --evaluate
.venv/bin/python -m src.q3_plotting
```

The `scp` example copies `SDPx_N3` and `SDPx_N7`; replace or add run names there to copy different runs. Evaluating
on the PC keeps all runs in one table and one set of figures even when some were solved on the PC and some on the
cluster. (The cluster's evaluation job does the same with the runs that are on the cluster.)

## Settings (`hpc/config.sh`)

| Setting | Default | Used for |
|---|---|---|
| `GUROBI_MODULE`, `PYTHON_MODULE` | `gurobi/13.0.0`, `python3/3.12.11` | HPC modules; Gurobi has the same major version as on the PC (13.0.3) |
| `CORES` | 8 | Cores per run, passed to Gurobi as `--threads` |
| `MEM_PER_CORE` | 4GB | LSF memory per core |
| `WALL` | 01:30 | Question 3 wall time per job (900 s start phase + 1800 s solve + margin) |
| `QUEUE` | hpc | LSF queue |
| `Q2_TIME_LIMIT` | 3600 | Seconds per Question 2 DP variant |
| `Q2_WALL` | 03:30 | Question 2 job wall time, enough for three variants plus setup/reporting |

The Question 3 run time limits are in `RUNS` of `src/q3_analysis.py`; raise `WALL` with them. Adjust
`Q2_TIME_LIMIT` and `Q2_WALL` for Question 2 as needed.
