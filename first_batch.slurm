#!/bin/bash -l

####################################
#     SLURM Job Submission Script  #
#                                  #
# Submit script: sbatch submit_job.slurm #
#                                  #
####################################

#SBATCH --job-name=parallel_test_variables # Job name
#SBATCH --nodes=1                         # Number of nodes requested
#SBATCH --ntasks=1                        # One task per node for GNU Parallel
#SBATCH --cpus-per-task=5                # Number of CPUs per task - align with node CPU count
#SBATCH --time=48:00:00                   # Walltime
#SBATCH --mem=14G                          # memory per NODE
#SBATCH --partition=el7thin               # Partition
#SBATCH --account=pa240701                # Replace with your system project

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
export C_INCLUDE_PATH=$HDF5_DIR/include/$C_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=$HDF5_DIR/include/$C_INCLUDE_PATH
export MPI_DIR=/apps/compilers/intel/18.0.4/impi/2018.4.274
export TMPDIR=/dev/shm/kpanag

export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8
# Define the function to run for each year-month pair
process_pair() {
    YEAR=$1
    MONTH=$2
    ORDER=1  # Example: set to 1 for the first trading day, 2 for the second, etc.
        
    if [ -z "$SLURM_JOB_ID" ]; then
        echo "Warning: SLURM_JOB_ID not set, using default" >&2
        SLURM_JOB_ID="default"
    fi
    # Find the specified trading day for the specified year and month
    TRADING_DAY_OUTPUT=$(python3.11 get_trading_day.py ${YEAR} ${MONTH} ${ORDER})
    VALID=$(echo "$TRADING_DAY_OUTPUT" | head -n 1)
    DAY=$(echo "$TRADING_DAY_OUTPUT" | tail -n 1)

    # Check if the Python script returned "valid=TRUE"
    if [ "$VALID" != "TRUE" ]; then
        echo "No valid trading day found for ${YEAR}-${MONTH} with order $ORDER"
        return
    fi
    
    OUTPUT_FILE="/work/pa24/kpanag/output_${YEAR}_${MONTH}.txt"
    touch $OUTPUT_FILE 
    echo ${YEAR}-${MONTH}-${DAY} > $OUTPUT_FILE
     
    # Run the execution script with the found trading day
    bash /work/pa24/kpanag/develop_scripts/execution.sh HPC $YEAR $MONTH -- $DAY -- all > $OUTPUT_FILE 2>&1
}

export -f process_pair

# Read the year-month pairs from the file and process them in parallel
cat /work/pa24/kpanag/develop_scripts/output_files/batch_11.txt | parallel -j $SLURM_CPUS_PER_TASK --colsep ' ' process_pair {1} {2}

