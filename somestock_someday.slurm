#!/bin/bash -l

####################################
#     SLURM Job Submission Script  #
#                                  #
# Submit script: sbatch submit_job.slurm #
#                                  #
####################################

#SBATCH --job-name=first_test_variables # Job name
#SBATCH --nodes=1                      # Number of nodes requested
#SBATCH --ntasks=1                     # One task per node for GNU Parallel
#SBATCH --cpus-per-task=1              # Number of CPUs per task - align with node CPU count
#SBATCH --time=24:00:00                 # Walltime
#SBATCH --mem=3G                     # memory per NODE
#SBATCH --partition=el7thin           # Partition
#SBATCH --account=pa240701             # Replace with your system project


# Set up the environment
if [ -z "${SLURM_CPUS_PER_TASK+x}" ]; then
    export OMP_NUM_THREADS=1
else
    export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
fi

module purge
cd /work/pa24/kpanag/develop_scripts
source /work/pa24/kpanag/myenv2/bin/activate

export HDF5_DIR=/apps/libraries/hdf5/1.12.1/gnu
export PATH=$HDF5_DIR/bin:$PATH
export LD_LIBRARY_PATH=$HDF5_DIR/lib:$HDF5_DIR/lib64:$LD_LIBRARY_PATH
export LIBRARY_PATH=$HDF5_DIR/lib:$HDF5_DIR/lib64:$LIBRARY_PATH
export C_INCLUDE_PATH=$HDF5_DIR/include:$C_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=$HDF5_DIR/include:$CPLUS_INCLUDE_PATH
export MPI_DIR=/apps/compilers/intel/18.0.4/impi/2018.4.274


bash /work/pa24/kpanag/develop_scripts/execution_somestock_somedays.sh HPC 2022 10 03 IBM
