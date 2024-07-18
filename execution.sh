#!/bin/bash
# This script sets paths and arguments to execute a python script for multiple stocks and days
#Syntax:
#The syntax for a not HPC machine is:  ./script.sh REG 2014 03 -- 02 -- IBM
#The syntax for HPC is:  ./script.sh HPC 2014 03 -- 02 -- IBM 
#The syntax for one day is:  ./script.sh REG 2014 03 -- 02 -- IBM
#The syntax for one stock is: ./script.sh REG 2014 03 -- 02 -- IBM MSFT
# The syntax for some days is : ./script.sh REG 2014 03 -- 02 03 -- IBM
#The syntax for some stocks is : ./script.sh REG 2014 03 -- 02 03 -- IBM A AA ABC MSFT
#The syntax for all trading days for every stock is : ./script.sh REG 2014 03 -- all -- IBM A AA ABC MSFT
#The first three arguments are always the machine year month, these can change as well
#
#Modify
#Line 56, provide the appropriate path of the HDF5 for reading
#Lines 57-65, the paths of the remaining files are set to the current directory where the program runs

set -eu
export RUST_BACKTRACE=full

# Get the arguments
machine=$1
year=$2
month=$(printf "%02d" $3)
shift 3

# Skip over the first '--' delimiter
shift

days=()
stocks=()

# Loop through arguments until we encounter the second '--'
while [[ $# -gt 0 && "$1" != '--' ]]; do
    if [[ "$1" != "all" ]]; then
        days+=("$1")
    else
        days=("all")
    fi
    shift
done

# Skip over the second '--' delimiter
shift

# Remaining arguments are stocks
stocks=("$@")

# Get the current directory
current_dir=$(pwd)

# Common filenames from github, across all machines
hdf5OriginalFileName="${year}${month}.h5"
pythonScriptName="variables_v4.py"
excludeStocksFileName="exclude_stocks.txt"
hdf5ContentsFileName="hdf_files_${year}${month}.txt"
prepareAnalysisName="preparation_analysis.txt"
variablesAnalysisName="variable_analysis.txt"
profilingAnalysisName="profiling_analysis.txt"
emptyVariablesName="empty_bars.txt"

# Determine the base path and other paths based on the path_type argument
if [ "$machine" == "REG" ]; then
    hdf5OriginalFilePath="/mnt/e/repository/data/taq/processed/raw_hdf5/"
    hdf5VariableFilePath="$current_dir/"
    pythonScriptPath="$current_dir/"
    excludeStocksFilePath="$current_dir/"
    hdf5ContentsFilePath="$current_dir/"
    prepareAnalysisPath="$current_dir/"
    variablesAnalysisPath="$current_dir/"
    profilingAnalysisPath="$current_dir/"
    emptyVariablesPath="$current_dir/"
elif [ "$machine" == "HPC" ]; then
    hdf5OriginalFilePath="/work/pa24/kpanag/output/"
    hdf5VariableFilePath="/work/pa24/kpanag/variables/"
    pythonScriptPath="/work/pa24/kpanag/develop_scripts/"
    excludeStocksFilePath="/work/pa24/kpanag/develop_scripts/"
    hdf5ContentsFilePath="/work/pa24/kpanag/hdf_check_byname/hdf_contents/"
    prepareAnalysisPath="/work/pa24/kpanag/performance/"
    variablesAnalysisPath="/work/pa24/kpanag/performance/"
    profilingAnalysisPath="/work/pa24/kpanag/performance/"
    emptyVariablesPath="/work/pa24/kpanag/performance/"
    hdf5ContentsFileName="hdf_files_${year}${month}"
else
    echo "Invalid path type. Use 'REG' or 'HPC'."
    exit 1
fi

# Read excluded stocks into an array
excludeStocksFile="${excludeStocksFilePath}${excludeStocksFileName}"
exclude_stocks=()
while IFS= read -r stock; do
    exclude_stocks+=("$stock")
done < "$excludeStocksFile"

# Define the files
hdf5OriginalFile="${hdf5OriginalFilePath}${hdf5OriginalFileName}"
hdf5ContentsFile="${hdf5ContentsFilePath}${hdf5ContentsFileName}"
pythonScript="${pythonScriptPath}${pythonScriptName}"
prepareAnalysis="${prepareAnalysisPath}${prepareAnalysisName}"
variablesAnalysis="${variablesAnalysisPath}${variablesAnalysisName}"
profilingAnalysis="${profilingAnalysisPath}${profilingAnalysisName}"
emptyVariables="${emptyVariablesPath}${emptyVariablesName}"

# Find the contents of the HDF5 file to find all trading stocks
if [ -f "$hdf5OriginalFile" ]; then
    if [ ! -f "$hdf5ContentsFile" ]; then
        h5ls -r $hdf5OriginalFile > "$hdf5ContentsFile"
    fi

    # Find all trading stocks from the contents
    IFS=$'\n'
    available_stocks=($(grep '/day[0-9][0-9]/ctm/table' "$hdf5ContentsFile" | sed -E 's|/([^/]+)/day[0-9][0-9]/ctm/table.*|\1|' | sort | uniq))
    unset IFS

    # Ignore non-valid stocks based on TAQ Cleaning techniques page 3
    if [ "${stocks[0]}" == "all" ]; then
        stocks=("${available_stocks[@]}")
    fi

    for stock in "${stocks[@]}"; do
        if [[ " ${exclude_stocks[@]} " =~ " ${stock} " ]]; then
            echo "Stock $stock is in the exclude list, skipping..."
            continue
        fi

        # Find all available days from the contents
        IFS=$'\n'
        available_days=($(grep "/${stock}/day[0-9][0-9]/ctm/table" "$hdf5ContentsFile" | sed -E 's|.*/day([0-9][0-9])/ctm/table.*|\1|' | sort | uniq))
        unset IFS

        if [ ${#available_days[@]} -eq 0 ]; then
            echo "No transactions found for stock $stock in month ${month}-${year}"
            continue
        fi

        if [ "${days[0]}" == "all" ]; then
            days=("${available_days[@]}")
        fi

        for available_day in "${days[@]}"; do
            hdf5VariableFileName="${year}${month}${available_day}_variables.h5"
            hdf5VariableFile="${hdf5VariableFilePath}${hdf5VariableFileName}"
            date_str="${year}-${month}-${available_day}"
            ctm_dataset_path="/${stock}/day${available_day}/ctm/table"
            complete_nbbo_dataset_path="/${stock}/day${available_day}/complete_nbbo/table"
            echo "Executing: $pythonScript $hdf5OriginalFile $date_str $stock $year $month $available_day"
            python3.11 $pythonScript $hdf5OriginalFile $date_str $stock $year $month $available_day $ctm_dataset_path $complete_nbbo_dataset_path $hdf5VariableFile --prep_analysis_path $prepareAnalysis $emptyVariables --var_analysis_path $variablesAnalysis --prof_analysis_path $profilingAnalysis
            echo "Executed for: $date_str, Stock: $stock, HDF5: $hdf5OriginalFile"
        done
    done
else
    echo "No HDF5 file found for $year-$month, skipping..."
fi
