#!/bin/bash
set -eu
export RUST_BACKTRACE=full
script_path="variables_v4.py"
hdf5_base_path="/home/taq/taq_variables"
stocks=("IBM")
exclude_stocks_file="/home/taq/taq_variables/exclude_stocks.txt"

# Read excluded stocks into an array
exclude_stocks=()
while IFS= read -r stock; do
    exclude_stocks+=("$stock")
done < "$exclude_stocks_file"

for year in {2009..2009}; do
    for month in {03..03}; do
        formatted_month=$(printf "%02d" $month)
        hdf5_file_path="${hdf5_base_path}/${year}${formatted_month}.h5"

        if [ -f "$hdf5_file_path" ]; then
            for stock in "${stocks[@]}"; do
                # Check if stock is in the exclude list
                if [[ " ${exclude_stocks[@]} " =~ " ${stock} " ]]; then
                    echo "Stock $stock is in the exclude list, skipping..."
                    continue
                fi

                for day in {02..02}; do
                    day_index=$(printf "day%02d" $day)
                    date_str="${year}-${formatted_month}-${day}"
                    ctm_dataset_path="/${stock}/${day_index}/ctm/table"
                    complete_nbbo_dataset_path="/${stock}/${day_index}/complete_nbbo/table"
                    echo "Executing: ./$script_path $hdf5_file_path $date_str $stock $year $formatted_month $day"
                    ./$script_path $hdf5_file_path $date_str $stock $year $formatted_month $day $ctm_dataset_path $complete_nbbo_dataset_path
                    echo "Executed for: $date_str, Stock: $stock, Year: $year, Month: $formatted_month, Day: $day, HDF5: $hdf5_file_path"
                done
            done
        else
            echo "No HDF5 file found for $year-$formatted_month, skipping..."
        fi
    done
done
