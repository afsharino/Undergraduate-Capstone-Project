# Import libraries
import os
import json

# A function for reading datasets
def read_data(BASE_DIR):
    data = dict()
    for dir in os.listdir(BASE_DIR):
        if 'csv' in dir or '.ipynb_checkpoints' in dir:
            continue
        print(f"\n\n\nEnter ____ {dir} ____ Directory")
        data[dir] = {}
        dir_path = os.path.join(BASE_DIR, dir)
        for file_path in glob.glob(os.path.join(dir_path, '*.json')):
            name = file_path.split('\\')[-1].split(".")[0].strip()
            print(f"Start Reading {name}...")
             # Load JSON file into DataFrame
            with open(file_path, 'r') as file:
                # Load json file
                json_data = json.load(file)['data']
                data[dir][name] = json_data
                print(f"{name} loaded succesfully :)\n")
    return data