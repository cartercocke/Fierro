#!/bin/bash
#SBATCH --time=6:00:00   # walltime
#SBATCH --ntasks=1   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --cpus-per-task=32   # number of CPU cores per task

module load gcc/13.2.0-gcc-13.2.0-w55nxkl 
module load cmake/3.27.9-gcc-13.2.0-g5dukmj    
module load openmpi/5.0.1-gcc-13.2.0-xpoh5uw

# bash build_evpfft.sh --heffte_build_type=fftw --kokkos_build_type=openmp --build_fftw --build_hdf5 --machine=linux  

python3 -u mpi_cpp_runner.py

# salloc -N 1 -n 1 --cpus-per-task=32 --time=4:00:00