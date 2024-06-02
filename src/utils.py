# Import libraries
import os
import json
import glob

#_____________ read_data _____________
def read_data(base_dir:str) -> dict:
    """A function for reading datasets

    Args:
        base_dir (str): directory that contains sub-directories of datasets

    Returns:
        dict: a dictionary of data  key(dir_name) : value(key(indiactor_name) : value(indicator_Values))
    """
    data = dict()

    # Check if the directory is empty or not
    if os.listdir(base_dir) == []:
        print(f"The directory is empty!\n")
        return

    for dir_name in os.listdir(base_dir):
        # do not read csv file which contains fear and greed indicator
        if 'csv' in dir_name:
            continue

        # Enter directories and read indicators one by one
        print(f"\n\n\nEnter ____ {dir_name} ____ Directory")
        data[dir_name] = {}
        dir_path = os.path.join(base_dir, dir_name)
        for file_path in glob.glob(os.path.join(dir_path, '*.json')):
            name = file_path.split('/')[-1].split(".")[0].strip() # Change "/" to \\ if you are on windows!
            print(f"Start Reading {name}...")
             # Load JSON file into DataFrame
            with open(file_path, 'r') as file:
                # Load json file
                json_data = json.load(file)['data']
                data[dir_name][name] = json_data
                print(f"{name} loaded succesfully :)\n")

    return data