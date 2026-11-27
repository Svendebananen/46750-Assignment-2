#!/bin/bash
# Submit the Question 3 runs to LSF, one job per run, each started when the run it starts from is done:
#   bash hpc/submit_q3.sh                    # every run of src/q3_analysis.RUNS that is not solved yet
#   bash hpc/submit_q3.sh SDPx_N3 SDPx_N7    # only these (their start runs must be solved or submitted too)
# A last job evaluates every run out of sample and draws figures 2-4.  Logs: hpc/logs/.
set -euo pipefail
cd "$(dirname "$0")/.."
source hpc/config.sh
module load "$PYTHON_MODULE" "$GUROBI_MODULE"
source "$VENV/bin/activate"
mkdir -p hpc/logs

# "name start solved" for every run, in the order of RUNS
mapfile -t ROWS < <(python - "$@" <<'PY'
import sys
from src.q3_analysis import RUNS, RESULTS_DIR, run_done
wanted = sys.argv[1:] or list(RUNS)
for name, cfg in RUNS.items():
    if name in wanted:
        print(name, cfg.get("start", "-"), int(run_done(RESULTS_DIR / "base" / "q3" / "runs" / name)))
PY
)

submitted=()
for row in "${ROWS[@]}"; do
    read -r name start solved <<< "$row"
    if [[ "$solved" == 1 ]]; then echo "$name: already solved, skipped"; continue; fi
    dep=""
    if [[ " ${submitted[*]} " == *" $start "* ]]; then dep="#BSUB -w \"done(q3_$start)\""; fi
    bsub <<JOB
#!/bin/bash
#BSUB -J q3_$name
#BSUB -q $QUEUE
#BSUB -n $CORES
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=$MEM_PER_CORE]"
#BSUB -W $WALL
#BSUB -o hpc/logs/q3_${name}_%J.out
#BSUB -e hpc/logs/q3_${name}_%J.err
$dep
module load $PYTHON_MODULE $GUROBI_MODULE
source $VENV/bin/activate
python -m src.q3_analysis --runs --only $name --threads $CORES
JOB
    submitted+=("$name")
    echo "$name: submitted${dep:+ (after $start)}"
done

if (( ${#submitted[@]} )); then
    cond=$(printf 'done(q3_%s) && ' "${submitted[@]}"); cond=${cond% && }
    bsub <<JOB
#!/bin/bash
#BSUB -J q3_evaluate
#BSUB -q $QUEUE
#BSUB -n 2
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=$MEM_PER_CORE]"
#BSUB -W 00:30
#BSUB -o hpc/logs/q3_evaluate_%J.out
#BSUB -e hpc/logs/q3_evaluate_%J.err
#BSUB -w "$cond"
module load $PYTHON_MODULE $GUROBI_MODULE
source $VENV/bin/activate
python -m src.q3_analysis --evaluate --threads 2
python -m src.q3_figures
JOB
    echo "q3_evaluate: submitted (after all runs)"
fi
