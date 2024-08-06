#!/usr/bin/env python3.11
# This file creates small copies of the original hdf5 files for the specified stocks and the specified days.
# To call this script please make sure of the appropriate directories in lines 23, 24
# The syntax is:
# python3.11 copy_hdf5.py year month -- days NUMBER -- stocks NAME
# python3.11 copy_hdf5.py 2018 10 --days 03 04 05 --stocks all

import h5py
import argparse

# Custom parser to handle -- separator
class CustomArgumentParser(argparse.ArgumentParser):
    def convert_arg_line_to_args(self, arg_line):
        return arg_line.split()

# Set up argument parser
parser = CustomArgumentParser(description='Process HDF5 files.')
parser.add_argument('year', type=str, help='Year of the data')
parser.add_argument('month', type=str, help='Month of the data')
parser.add_argument('--days', nargs='+', help='Days of the data', required=True)
parser.add_argument('--stocks', nargs='+', help='Stock symbols', required=True)

args = parser.parse_args()
year = args.year
month = args.month
days = args.days
stocks = args.stocks

original_file_path = f'/work/pa24/kpanag/output/{year}{month}.h5'
new_file_path = f'/dev/shm/kpanag/{year}{month}.h5'

# If stocks are "all", find all available stocks for the specified days
if "all" in stocks:
    hdf5ContentsFilePath = "/work/pa24/kpanag/hdf_check_byname/hdf_contents/"
    hdf5ContentsFileName = f"hdf_files_{year}{month}.txt"
    hdf5ContentsFile = f"{hdf5ContentsFilePath}{hdf5ContentsFileName}"

    # Read the available stocks from the contents file
    available_stocks = []
    with open(hdf5ContentsFile, 'r') as file:
        lines = file.readlines()
        for day in days:
            day_pattern = f'/day{day}/ctm/table'
            for line in lines:
                if day_pattern in line:
                    stock = line.split('/')[1]  # Extract the stock name from the path
                    if stock not in available_stocks:
                        available_stocks.append(stock)
else:
    available_stocks = stocks

# List of datasets to copy
datasets_to_copy = []

for day in days:
    for stock in available_stocks:
        datasets_to_copy.append(f'/{stock}/day{day}/ctm/')
        datasets_to_copy.append(f'/{stock}/day{day}/complete_nbbo/')

def ensure_group(h5file, group_path):
    """ Ensure all groups in the given path exist in the target file """
    parts = group_path.strip('/').split('/')
    for i in range(1, len(parts) + 1):
        path = '/' + '/'.join(parts[:i])
        if path not in h5file:
            h5file.create_group(path)

def copy_dataset(source, target):
    """ Copy dataset from source to target """
    target[...] = source[...]

# Open the original file
with h5py.File(original_file_path, 'r') as original_file:
    # Create a new HDF5 file
    with h5py.File(new_file_path, 'w') as new_file:
        for group_path in datasets_to_copy:
            # Check if the group exists in the original file
            if group_path in original_file:
                print(f"Found group: {group_path}")
                # Ensure all groups leading to the dataset exist
                ensure_group(new_file, group_path)

                # Iterate through datasets within the group
                for dataset_name in original_file[group_path].keys():
                    dataset_to_copy = f"{group_path}{dataset_name}"
                    if dataset_to_copy in original_file:
                        print(f"Found dataset: {dataset_to_copy}")

                        # Read the dataset from the original file
                        source_dataset = original_file[dataset_to_copy]

                        # Create the dataset in the new file with the same shape and dtype
                        target_dataset = new_file.create_dataset(dataset_to_copy, shape=source_dataset.shape, dtype=source_dataset.dtype)

                        # Copy dataset content
                        copy_dataset(source_dataset, target_dataset)

                        # Copy the attribute 'column_names' if it exists on the group level
                        group = original_file[group_path]
                        if 'column_names' in group.attrs:
                            new_file[group_path].attrs['column_names'] = group.attrs['column_names']
                            print(f"Copied attribute 'column_names': {group.attrs['column_names']}")

                        print(f"Copied dataset {dataset_to_copy}")

                    else:
                        print(f"Dataset path {dataset_to_copy} does not exist in the original file.")

            else:
                print(f"Group path {group_path} does not exist in the original file.")

print(f"New HDF5 file created at {new_file_path}")
