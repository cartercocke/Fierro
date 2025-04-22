#!/bin/bash
#SBATCH --time=6:00:00   # walltime
#SBATCH --ntasks=1   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --cpus-per-task=32   # number of CPU cores per task

python3 -u mpi_cpp_runner.py