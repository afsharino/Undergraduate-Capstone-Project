# Import libraries
import os
import json
import glob
import pandas as pd

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

#_____________ convert_data_to_dataframe _____________
def convert_data_to_dataframe(data:dict) -> dict:
    """This function used to convert indicator's data and timestap to dataframe

    Args:
        data (dict): a dictionary of data  key(dir_name) : value(key(indiactor_name) : value(indicator_Values))

    Returns:
        dict: a dictionary of data  key(dir_name) : value(list[Pandas_DataFrames])
    """
    all_dataframes = {}
    for directory in data.keys():
        temp_df = []
        for indicator in data[directory].keys():
            # Convert loaded json to pandas dataframe
            temp_df_indicator = pd.DataFrame(data[directory][indicator], columns=['Timestamp', indicator])
            
            # Convert Unix timestamp to datetime
            temp_df_indicator['Date'] = pd.to_datetime(temp_df_indicator['Timestamp'], unit='ms')
    
            # # Drop the 'Timestamp' column
            temp_df_indicator.drop(columns=['Timestamp'], inplace=True)
            
            # Append the DataFrame to our list of DataFrames
            temp_df.append(temp_df_indicator)
    
        all_dataframes[directory] = temp_df

    return all_dataframes