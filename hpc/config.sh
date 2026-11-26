# Settings for DTU HPC jobs. Check module names with:  module avail gurobi  /  module avail python3
GUROBI_MODULE="gurobi/13.0.0"      # same major version as on the PC (13)
PYTHON_MODULE="python3/3.12.11"    # the Python of the gurobipy 13.0.0 module
VENV=".venv-hpc"                   # virtual environment in the repository folder
QUEUE="hpc"                        # LSF queue
CORES=8                            # cores per run (passed to Gurobi as --threads)
MEM_PER_CORE="4GB"                 # LSF rusage is per core
WALL="01:30"                       # wall time per run (hh:mm): 900 s start + 1800 s solve + margin
Q2_TIME_LIMIT=3600                 # seconds per DP variant; --DP --ir solves three variants
Q2_WALL="03:30"                    # enough for three Q2 variants plus setup/reporting
