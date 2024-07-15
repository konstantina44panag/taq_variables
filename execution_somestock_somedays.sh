#!/bin/bash
set -eu
export RUST_BACKTRACE=full

# Check if the correct number of arguments is passed
if [ "$#" -lt 5 ]; then
    echo "Usage: $0 <path_type> <year> <month> <day> <stock1> [<stock2> ... <stockN>]"
    exit 1
fi

# Get the arguments
path_type=$1
year=$2
month=$(printf "%02d" $3)
day=$(printf "%02d" $4)
shift 4
stocks=("$@")

# Determine the base path based on the path_type argument
if [ "$path_type" == "ATOM" ]; then
    hdf5_file_path="/mnt/e/wrds23_hdf5/output/${year}${month}.h5"
    hdf5_variable_path="${year}${month}_variables.h5"
    script_path="variables_v4.py"
    exclude_stocks_file="exclude_stocks.txt"
    contents_file="contents_${year}${month}.txt"
    prep_analysis_path="onedataset_preparation_analysis.txt"
    emp_analysis_path="onedataset_empty_bars.txt"
    var_analysis_path="onedataset_variable_analysis.txt"
    prof_analysis_path="onedataset_profiling_analysis.txt"
elif [ "$path_type" == "HPC" ]; then
    hdf5_file_path="/work/pa24/kpanag/output/${year}${month}.h5"
    hdf5_variable_path="/work/pa24/kpanag/variables/${year}${month}_variables.h5"
    script_path="/work/pa24/kpanag/develop_scripts/variables_v4.py"
    exclude_stocks_file="/work/pa24/kpanag/develop_scripts/exclude_stocks.txt"
    contents_file="/work/pa24/kpanag/hdf_check_byname/hdf_contents/hdf_files_${year}${month}"
    prep_analysis_path="/work/pa24/kpanag/performance/onedataset_preparation_analysis.txt"
    emp_analysis_path="/work/pa24/kpanag/performance/onedataset_empty_bars.txt"
    var_analysis_path="/work/pa24/kpanag/performance/onedataset_variable_analysis.txt"
    prof_analysis_path="/work/pa24/kpanag/performance/onedataset_profiling_analysis.txt"
else
    echo "Invalid path type. Use 'ATOM' or 'HPC'."
    exit 1
fi

# Read excluded stocks into an array
exclude_stocks=()
while IFS= read -r stock; do
    exclude_stocks+=("$stock")
done < "$exclude_stocks_file"

#Ignore non valid stocks based on TAQ Cleaning techniques page.3 
#For every stock-day send the necessary information to the python scripts responsible for loading, cleaning and computing 
if [ -f "$hdf5_file_path" ]; then
    for stock in "${stocks[@]}"; do
        if [[ " ${exclude_stocks[@]} " =~ " ${stock} " ]]; then
            echo "Stock $stock is in the exclude list, skipping..."
            continue
        fi
        date_str="${year}-${month}-${day}"
        ctm_dataset_path="/${stock}/day${day}/ctm/table"
        complete_nbbo_dataset_path="/${stock}/day${day}/complete_nbbo/table"
        echo "Executing: ./$script_path for $hdf5_file_path $date_str $stock"
        python3.11 $script_path $hdf5_file_path $date_str $stock $year $month $day $ctm_dataset_path $complete_nbbo_dataset_path $hdf5_variable_path $prep_analysis_path $emp_analysis_path $var_analysis_path $prof_analysis_path
        echo "Executed for: $date_str, Stock: $stock, Year: $year, Month: $month, Day: $day, HDF5: $hdf5_file_path"
    done
else
    echo "No HDF5 file found for $year-$month, skipping..."
fi
