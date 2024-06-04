# Import libraries
import os
import json
import glob
import pandas as pd
import numpy as np
from typing import List

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

#_____________ check_date_integrity _____________
def check_date_integrity(dfs:List[pd.DataFrame], directory_name:str) -> None:
    """A function to check integrity between dates in each category

    Args:
        dfs (List[pd.DataFrame]): list of indicator DataFrame
        directory_name (str): Main Category that contains indicators
    """
    try:
        # Get the Date column of the first DataFrame
        reference_dates = dfs[0]['Date']
        
        # Flag to indicate if all Date columns are the same
        all_dates_same = True
        
        # Iterate over the rest of the DataFrames
        for df in dfs[1:]:
            # Check if the Date column of the current DataFrame is equal to the reference Dates
            if not np.all(df['Date'] == reference_dates):
                all_dates_same = False
                break
        
        if all_dates_same:
            print(f"All DataFrames have the same Date values for '{directory_name}' directory.\n")
        else:
            print(f"Not all DataFrames have the same Date values for '{directory_name}' directory.\n")
    except Exception as e:
        print(f"Error: {e} for {directory_name}\n")


#_____________ check_shape_integrity _____________
def check_shape_integrity(dfs:List[pd.DataFrame], directory_name:str) -> None:
    """A Function to check for shape integrity between indicators in one category

    Args:
        dfs (List[pd.DataFrame]): list of indicator DataFrame
        directory_name (str): Main Category that contains indicators
    """
    try:

        # Get the shape of the first DataFrame
        reference_shape = dfs[0].shape[0]

        # Get the name of the first indicator
        reference_indicator_name = dfs[0].columns[0]

        # Current indicator's name
        current_indicator_name = ""

        # Flag to indicate if all shapes are the same
        all_shapes_same = True

        # Iterate over the rest of the DataFrames
        for df in dfs[1:]:
            # Get the shape of the current DataFrame
            current_shape = df.shape[0]

            # Get the name of the current indicator
            current_indicator_name = df.columns[0]

            # Check if the  shape of the first column of the current DataFrame is equal to the reference shape
            if not (df.shape[0] == reference_shape):
                all_shapes_same = False
                break

        if all_shapes_same:
            print(f"All DataFrames have the same shape values for '{directory_name}' directory.")
            print(f"The shape is {reference_shape}.\n")
        else:
            print(f"Not all DataFrames have the same Date values for '{directory_name}' directory.")
            print(f"Refrence: {reference_indicator_name}:{reference_shape} - Current:{current_indicator_name}:{current_shape}\n")
    except Exception as e:
        print(f"Error: {e} for {directory_name}-{current_indicator_name}\n")

#_____________ check_period_integrity _____________
def check_period_integrity(dfs:List[pd.DataFrame], directory_name:str) -> None:
    """A Function to show start and end date of the each indicator

    Args:
        dfs (List[pd.DataFrame]): list of indicator DataFrame
        directory_name (str): Main Category that contains indicators
    """
    print(f"Period Integrity for _______ {directory_name} _______\n")
    for indicator in dfs:
        print((str(indicator.Date.min()).split()[0], str(indicator.Date.max()).split()[0]))

#_____________ drop_indicator _____________
def drop_indicator(dfs:List[pd.DataFrame], based_on:str, month:str=None, day:str=None) -> list:
    """The function is used to drop the indicators that can not be aligend with other indicators because of less datapoints

    Args:
        dfs (List[pd.DataFrame]): list of indicator DataFrame
        based_on (str): _shoud be chosen between on of the "start" or "end"
        month (str, optional): Between this parameter and day one should be chosen with value like "08". Defaults to None.
        day (str, optional): Between this parameter and month one should be chosen with value like "08". Defaults to None.

    Returns:
        list: return list of DataFrames that are alignable
    """
    if day == None:
        if based_on == 'start':
            return [indicator for indicator in dfs if month != (str(indicator.Date.min()).split()[0]).split('-')[1]]
        
        elif based_on == 'end':
            return [indicator for indicator in dfs if month != (str(indicator.Date.max()).split()[0]).split('-')[1]]
        else:
            print(f"Invalid based_on parameter!")

    if month == None:
        if based_on == 'start':
            return [indicator for indicator in dfs if day != (str(indicator.Date.min()).split()[0]).split('-')[2]]
        
        elif based_on == 'end':
            return [indicator for indicator in dfs if day != (str(indicator.Date.max()).split()[0]).split('-')[2]]
        else:
            print(f"Invalid based_on parameter!")

    if day == None and month==None:
        print(f"Please send a value for one of the parameters month or day!")
    
    if day != None and month!=None:
        print(f"You can not choose value for both month and day, one should be None!")


        
