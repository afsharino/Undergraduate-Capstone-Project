# Import libraries
import os
import json
import glob
import pandas as pd
import numpy as np
from typing import List, Tuple
from optimizers import linear_genetic_algorithm

#__________________________ read_data __________________________
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

#__________________________ convert_data_to_dataframe __________________________
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

#__________________________ check_date_integrity __________________________
def check_date_integrity(dfs:List[pd.DataFrame], directory_name:str) -> None:
    """A function to check integrity between dates in each category

    Args:
        dfs (List[pd.DataFrame]): list of indicator DataFrames
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


#__________________________ check_shape_integrity __________________________
def check_shape_integrity(dfs:List[pd.DataFrame], directory_name:str) -> None:
    """A Function to check for shape integrity between indicators in one category

    Args:
        dfs (List[pd.DataFrame]): list of indicator DataFrames
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

#__________________________ check_period_integrity __________________________
def check_period_integrity(dfs:List[pd.DataFrame], directory_name:str) -> None:
    """A Function to show start and end date of the each indicator

    Args:
        dfs (List[pd.DataFrame]): list of indicator DataFrames
        directory_name (str): Main Category that contains indicators
    """
    print(f"Period Integrity for _______ {directory_name} _______\n")
    for indicator in dfs:
        print((str(indicator.Date.min()).split()[0], str(indicator.Date.max()).split()[0]))

#__________________________ drop_indicator __________________________
def drop_indicator(dfs:List[pd.DataFrame], based_on:str, month:str=None, day:str=None) -> List[pd.DataFrame]:
    """The function is used to drop the indicators that can not be aligend with other indicators because of less datapoints

    Args:
        dfs (List[pd.DataFrame]): list of indicator DataFrames
        based_on (str): _shoud be chosen between on of the "start" or "end"
        month (str, optional): Between this parameter and day one should be chosen with value like "08". Defaults to None.
        day (str, optional): Between this parameter and month one should be chosen with value like "08". Defaults to None.

    Returns:
        List[pd.DataFrame]: return list of DataFrames that are alignable
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

#__________________________ align_dataframes __________________________
def align_dataframes(dfs:List[pd.DataFrame], directory_name:str) -> List[pd.DataFrame]:
    """A f unction to align all indicators to same time period

    Args:
        dfs (List[pd.DataFrame]): list of indicator DataFrames
        directory_name (str): Main Category that contains indicators

    Returns:
        List[pd.DataFrame]: return list of DataFrames that are aliged
    """
    for index in range(len(dfs)):     
        dfs[index] = dfs[index][(dfs[index]['Date'] >= '2021-03-12') & (dfs[index]['Date'] <= '2024-03-08')].copy()
        dfs[index].reset_index(inplace=True)
        dfs[index].drop(columns=['index'], inplace=True)

    print(f"{directory_name} DataFrames aligned succeessfully :)")

    return dfs

#__________________________ combine_dfs __________________________
def combine_dfs(df:List[pd.DataFrame]) -> pd.DataFrame:
    """A function used to combine indicator DataFrames in each category

    Args:
        df (List[pd.DataFrame]): list of indicator DataFrames

    Returns:
        pd.DataFrame: A single DataFrame obtained by concatenating the input DataFrames along the columns,
                  with any duplicate columns removed.

    Example:
    >>> df1 = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    >>> df2 = pd.DataFrame({'A': [5, 6], 'C': [7, 8]})
    >>> df_list = [df1, df2]
    >>> combined_df = remove_duplicate_columns(df_list)
    >>> print(combined_df)
       A  B  C
    0  1  3  7
    1  2  4  8
    """
    # Concatenate a list of DataFrames horizontally (column-wise)
    combined_df = pd.concat(df, axis=1)

    # Remove duplicate columns from the concatenated DataFrame
    combined_df = combined_df.iloc[:, ~combined_df.columns.duplicated()]

    # Return the resulting DataFrame with duplicates removed
    return combined_df

#__________________________ fearandgreed_integrity_check __________________________
def fearandgreed_integrity_check(fearandgreed_df:pd.DataFrame, others_df:pd.DataFrame, dir_name:str) -> None:
    """A function to check integrity between "fear and greed" dates and other indicators.

    Args:
        fearandgreed_df (pd.DataFrame): Dataframe contain fear and greed data
        others_df (pd.DataFrame): Dataframe of indicators in each category
        dir_name (str): Main Category that contains indicators
    """
    try:
        # Check if the Date column of the current DataFrame is equal to the fearandgreed Dates
        if not np.all(fearandgreed_df['Date'] == others_df['Date']):
            print(f"Not all DataFrames have the same Date values for '{dir_name}' directory.")
        else:
            print(f"All DataFrames have the same Date values for '{dir_name}' directory.")
    except Exception as e:
        print(f"Error: {e} for {dir_name}\n")


#__________________________ strategy __________________________
def strategy(indcator:np.ndarray, prices=np.ndarray) -> tuple:
    """This function is the implementation of our strategy to run on test data

    Args:
        indcator (np.ndarray): indicator values,(fearandgreed or new_indicator)
        price (np.ndarray): actual prices 

    Returns:
        tuple: tuple of values needed for visualization
    """
    INITIAL_BALANCE = 10000
    cash_balance = INITIAL_BALANCE
    bitcoin_amount = 0
    total_balance = cash_balance
    profit = 0

    indicator_values = indcator
    prices = prices
    profits = []
    cash = []
    bitcoin = []
    total = []

    for i in range(len(indicator_values)):
        # Create New Indicator
        indicator_value = indicator_values[i]
        
        price = prices[i]

        total_balance =  cash_balance + (bitcoin_amount*price)
        total.append(total_balance)
        
        bitcoin_amount = ((indicator_value/100) * total_balance) / price
        bitcoin.append(bitcoin_amount)
        
        cash_balance = total_balance - (bitcoin_amount*price)
        cash.append(cash_balance)
        
        profit = total_balance - INITIAL_BALANCE
        profits.append(profit)

    return prices, indicator_values, profits, cash, bitcoin, total

#__________________________ process_window __________________________
def process_window(args):
    i, all_indicators, combined_dataframe, num_individuals, num_genes, num_generations, mutation_rate, initial_population, parallel_type, num_processors, WINDOW_SIZE = args
    train_data = all_indicators[i:i+WINDOW_SIZE]
    prices = combined_dataframe.price[i:i+WINDOW_SIZE].values

    # Find Coefficients in past 6 months
    fitness_values, best_fitness_sliding_window, best_coefficients_sliding_window = linear_genetic_algorithm(
        data_param=train_data,
        prices_param=prices,
        num_individuals=num_individuals,
        num_genes=num_genes,
        num_generations=num_generations,
        mutation_rate=mutation_rate,
        initial_population=initial_population,
        parallel_type=parallel_type,
        num_processors=num_processors,
    )
    # print("*"*50)
    # print(f"i = {i}")
    # print(best_fitness_sliding_window)
    # plot_fitness(fitness_values, num_generations, "Linear Sliding window Model")
    # print("*"*50)
    # print('\n\n')
    # Create New Indicator
    
    new_indicator = np.dot(all_indicators[i:i+WINDOW_SIZE+1], best_coefficients_sliding_window)
    min_value = np.min(new_indicator)
    max_value = np.max(new_indicator)
    # Normalize new_indicator to range [0, 100]
    new_indicator = ((new_indicator - min_value) / (max_value - min_value)) * 100
    
    if i == 0:
        new_indicator_values = list(new_indicator)
    else:
        new_indicator_values = [new_indicator[-1]]

    return (i, new_indicator_values)