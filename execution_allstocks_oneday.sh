#!/bin/bash
set -eu
script_path="variables_v2.py"
hdf5_base_path="/mnt/e/wrds23_hdf5/output"

for year in {2009..2009}; do
    for month in {03..03}; do
        formatted_month=$(printf "%02d" $month)
        hdf5_file_path="${hdf5_base_path}/${year}${formatted_month}.h5"
        contents_file="contents.txt"

        if [ -f "$hdf5_file_path" ]; then
            if [ ! -f "$contents_file" ]; then
                h5ls -r $hdf5_file_path > "$contents_file"
            fi
            
            IFS=$'\n'
            available_stocks=($(grep '/day[0-9][0-9]/ctm/table' "$contents_file" | sed -E 's|/([^/]+)/day[0-9][0-9]/ctm/table.*|\1|' | sort | uniq))
            unset IFS

            for stock in "${available_stocks[@]}"; do
                IFS=$'\n'
                available_days=($(grep "/${stock}/day[0-9][0-9]/ctm/table" "$contents_file" | sed -E 's|.*/day([0-9][0-9])/ctm/table.*|\1|' | sort | uniq))
                unset IFS

                if [ ${#available_days[@]} -eq 0 ]; then
                    echo "No transactions found for stock in month ${month}-${year}"
                    continue
                fi

                for day_num in "${available_days[@]}"; do
                    if [[ "$day_num" == "02" ]]; then
                        date_str="${year}-${formatted_month}-${day_num}"
                        ctm_dataset_path="/${stock}/day${day_num}/ctm/table"
                        complete_nbbo_dataset_path="/${stock}/day${day_num}/complete_nbbo/table"
                        echo "Executing: ./$script_path $hdf5_file_path $date_str $stock $year $formatted_month $day_num"
                        ./$script_path $hdf5_file_path $date_str $stock $year $formatted_month $day_num $ctm_dataset_path $complete_nbbo_dataset_path
                        echo "Executed for: $date_str, Stock: $stock, Year: $year, Month: $formatted_month, Day: $day_num, HDF5: $hdf5_file_path"
                    fi
                done
            done
        else
            echo "No HDF5 file found for $year-$formatted_month, skipping..."
        fi
    done
done
