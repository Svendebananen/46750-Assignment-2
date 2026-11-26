#!/bin/bash
# One-time shared Q2/Q3 setup on the DTU HPC interactive node:  bash hpc/setup_q3.sh
# Creates the virtual environment and checks that Gurobi finds a licence.
set -euo pipefail
cd "$(dirname "$0")/.."
source hpc/config.sh
module load "$PYTHON_MODULE" "$GUROBI_MODULE"
python3 -m venv "$VENV"
source "$VENV/bin/activate"
pip install --upgrade pip
pip install "gurobipy==${GUROBI_MODULE#*/}" "numpy>=1.26" "pandas>=2.1" "matplotlib>=3.8"
python - <<'PY'
import gurobipy as gp
m = gp.Model()                                  # fails here if no licence is found
x = m.addVars(3000, lb=0)                       # larger than the size-limited pip licence (2000 variables)
m.setObjective(x.sum()); m.optimize()
print("Gurobi", gp.gurobi.version(), "licence OK")
PY
