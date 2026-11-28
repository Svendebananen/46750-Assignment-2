#!/bin/bash
# Submit Question 2 dynamic pricing to LSF:
#   bash hpc/submit_q2.sh
# Runs python main.py --q2-dp --q2-ir; each DP variant has the time limit in hpc/config.sh.
set -euo pipefail
cd "$(dirname "$0")/.."
source hpc/config.sh
mkdir -p hpc/logs

bsub <<JOB
#!/bin/bash
#BSUB -J q2_dp_ir
#BSUB -q $QUEUE
#BSUB -n $CORES
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=$MEM_PER_CORE]"
#BSUB -W $Q2_WALL
#BSUB -o hpc/logs/q2_dp_ir_%J.out
#BSUB -e hpc/logs/q2_dp_ir_%J.err
set -euo pipefail
cd "$PWD"
module load $PYTHON_MODULE $GUROBI_MODULE
source $VENV/bin/activate
python main.py --q2-dp --q2-ir --q2-time-limit $Q2_TIME_LIMIT --q2-threads $CORES
JOB

echo "q2_dp_ir: submitted (up to three ${Q2_TIME_LIMIT}s DP solves)"
