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
#Modify the directories in lines:
#Line 56, provides the appropriate path for reading the HDF5. The path filled is the appropriate on ATOM for reading a small copy of hdf5. 
#For the original hdf5 the folder "test_hdf5" should be changed to "raw_hdf5"
#Lines 57-65, the paths of the remaining files are set to the current directory where the program runs
#Lines 56 -65 can change directories
set -eu
export RUST_BACKTRACE=full

# Get the arguments
machine=$1
year=$2
month=$(printf "%02d" $((10#$3)))
shift 3

shift

days=()
stocks=()
specific_stock=""

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

# Loop through arguments until we encounter the third '--' or end of arguments
while [[ $# -gt 0 && "$1" != '--' ]]; do
    stocks+=("$1")
    shift
done

# Skip over the third '--' delimiter if it exists
if [[ $# -gt 0 ]]; then
    shift
fi

# If there is one more argument, it is the specific stock
if [[ $# -gt 0 ]]; then
    specific_stock="$1"
fi

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
getsuffixName="get_suffix.py"
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
    getsuffixPath="$current_dir/"
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
    getsuffixPath="/work/pa24/kpanag/develop_scripts/"
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
getsuffixScript="${getsuffixPath}${getsuffixName}"
# Find the contents of the HDF5 file to find all trading stocks
if [ -f "$hdf5OriginalFile" ]; then
    if [ ! -f "$hdf5ContentsFile" ]; then
        h5ls -r $hdf5OriginalFile > "$hdf5ContentsFile"
    fi

    # Find all trading stocks from the contents
    IFS=$'\n'
    if [ ${#days[@]} -eq 1 ] && [ "${days[0]}" != "all" ]; then
        day=${days[0]}
        available_stocks=($(grep "/day${day}/ctm/table" "$hdf5ContentsFile"| sed -E "s|/([^/]+)/day${day}/ctm/table.*|\\1|" | sort | uniq))
        if [[ -n "$specific_stock" ]]; then
            available_stocks=($(printf "%s\n" "${available_stocks[@]}" | awk -v stock="$specific_stock" 'BEGIN {found=0} {if ($0 >= stock) found=1} found')) #or >
        fi
    else
        available_stocks=($(grep '/day[0-9][0-9]/ctm/table' "$hdf5ContentsFile" | sed -E 's|/([^/]+)/day[0-9][0-9]/ctm/table.*|\1|' | sort | uniq))
    fi
    unset IFS

    # Create a list of invalid stocks
    if [ "${stocks[0]}" == "all" ]; then
        stocks=("${available_stocks[@]}")
    fi

    #For every stock argument:
    for stock in "${stocks[@]}"; do
        #Exclude stocks, google doc Taq Cleaning Techniques, page 3
        if [[ " ${exclude_stocks[@]} " =~ " ${stock} " ]]; then
            echo "Stock $stock is in the exclude list, skipping..."
            continue
        fi

        #If the argument is to execute all days, then  all days will be the list of the loop, otherwise the list of loop is the list of day arguments
        if [ "${days[0]}" == "all" ]; then
            IFS=$'\n'
            available_days=($(grep "/${stock}/day[0-9][0-9]/ctm/table" "$hdf5ContentsFile" | sed -E 's|.*/day([0-9][0-9])/ctm/table.*|\1|' | sort | uniq))
            unset IFS
            if [ ${#available_days[@]} -eq 0 ]; then
                echo "No transactions found for stock $stock in month ${month}-${year}"
                continue
            fi
            days=("${available_days[@]}")
        fi

        #For every day in the list of days:
        for day in "${days[@]}"; do
            #In case the list of the loop is a list from the arguments, then check that each day exists in the available days when the stock traded
            #The variable file name should be defined here, because it depends on the day
            hdf5VariableFileName="${year}${month}${day}_variables.h5"
            hdf5VariableFile="${hdf5VariableFilePath}${hdf5VariableFileName}"
            #Construct the paths inside the hdf5 file
            date_str="${year}-${month}-${day}"
            ctm_dataset_path="/${stock}/day${day}/ctm/"
            complete_nbbo_dataset_path="/${stock}/day${day}/complete_nbbo/"
            unique_suffix_values=$(python3.11 $getsuffixScript $hdf5OriginalFile $ctm_dataset_path)
            if [ "$unique_suffix_values" == "None" ]; then
                echo "The $stock was not traded on ${month}-${year}-${day}"
                continue
            fi
            IFS=',' read -r -a suffix_array <<< "$unique_suffix_values"
            for s in "${suffix_array[@]}"; do
                python3.11 $pythonScript $hdf5OriginalFile $date_str $stock $s $year $month $day $ctm_dataset_path $complete_nbbo_dataset_path $hdf5VariableFile --prep_analysis_path $prepareAnalysis --empty_variables_path $emptyVariables --var_analysis_path $variablesAnalysis --prof_analysis_path $profilingAnalysis
                echo "Executed for: $date_str, Stock: $stock, Suffix: $s, HDF5: $hdf5OriginalFile, Variables in $hdf5VariableFile"
            done
        done
    done
else
    echo "No HDF5 file found for $year-$month, skipping..."
fi
