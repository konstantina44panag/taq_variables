#!/bin/bash
set -eu
script_path="variables_v2.py"
hdf5_base_path="/mnt/e/wrds23_hdf5/output"
stock="IBM"

for year in {2009..2009}; do
    for month in {03..03}; do
        formatted_month=$(printf "%02d" $month)
        hdf5_file_path="${hdf5_base_path}/${year}${formatted_month}.h5"

        if [ -f "$hdf5_file_path" ]; then
            IFS=$'\n'
            available_days=($(h5ls -r $hdf5_file_path | grep "/${stock}/day[0-9][0-9]/ctm/table" | cut -d ':' -f 1))
            unset IFS

            for day_path in "${available_days[@]}"; do
                day_index=$(echo "$day_path" | grep -o 'day[0-9][0-9]')
                day_num=$(echo $day_index | sed 's/day//')
                date_str="${year}-${formatted_month}-${day_num}"
                ctm_dataset_path="/${stock}/day${day_num}/ctm/table"
                complete_nbbo_dataset_path="/${stock}/day${day_num}/complete_nbbo/table"
                echo "Executing: ./$script_path $hdf5_file_path $date_str $stock $year $formatted_month $day_num"
                ./$script_path $hdf5_file_path $date_str $stock $year $formatted_month $day_num $ctm_dataset_path $complete_nbbo_dataset_path
                echo "Executed for: $date_str, Stock: $stock, Year: $year, Month: $formatted_month, Day: $day_num, HDF5: $hdf5_file_path"
            done
        else
            echo "No HDF5 file found for $year-$formatted_month, skipping..."
        fi
    done
done
