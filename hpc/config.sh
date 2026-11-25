# Settings for the DTU HPC jobs of Question 3. Check the module names with:  module avail gurobi  /  module avail python3
GUROBI_MODULE="gurobi/13.0.3"      # same major version as gurobipy in the venv (13)
PYTHON_MODULE="python3/3.12.4"     # any Python >= 3.10
VENV=".venv-hpc"                   # virtual environment in the repository folder
QUEUE="hpc"                        # LSF queue
CORES=8                            # cores per run (passed to Gurobi as --threads)
MEM_PER_CORE="4GB"                 # LSF rusage is per core
WALL="01:30"                       # wall time per run (hh:mm): 900 s start + 1800 s solve + margin
