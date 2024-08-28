# Import libraries
import numpy as np
import pygad
import matplotlib.pyplot as plt
from tqdm import tqdm

#__________________________  Use Linear Combination as a Model __________________________
# Define the fitness function
def linear_fitness_function(ga_instance, solution, solution_idx):
    new_indicator = np.dot(data, solution)

    # Normalize new_indicator to range [0, 100]
    min_value = np.min(new_indicator)
    max_value = np.max(new_indicator)

    if min_value == max_value:
        # If min_value equals max_value, avoid normalization
        # Set all predictions to a fixed value
        print("***************************************")
        print(solution, end="\n")
        print("***************************************")
        
        predictions.fill(min_value)

    else:
        # Normalize new_indicator to range [0, 100]
        predictions = ((predictions - min_value) / (max_value - min_value)) * 100

    
    INITIAL_BALANCE = 10000
    cash_balance = INITIAL_BALANCE
    bitcoin_amount = 0
    total_balance = cash_balance
    profit = 0
    
    for i in range(len(prices)):
        indicator_value = new_indicator[i]
        price = prices[i]
        
        total_balance =  cash_balance + (bitcoin_amount * price)
        bitcoin_amount = ((indicator_value/100) * total_balance) / price
        cash_balance = total_balance - (bitcoin_amount * price)
        profit = total_balance - INITIAL_BALANCE

    return profit

def linear_on_generation(ga_instance):
    #global pbar
    linear_pbar.update(1)


def linear_genetic_algorithm(data_param, prices_param, num_individuals, num_genes, num_generations, mutation_rate, initial_population, parallel_type, num_processors, verbose=None):
    global data, prices
    
    # Create the initial population
    num_individuals = num_individuals
    num_genes = num_genes
    initial_population = initial_population

    # Configure Genetic Algorithm parameters
    num_generations = num_generations
    mutation_rate = mutation_rate

    # Data
    data = data_param
    prices = prices_param

    # Parallel Processing Setup
    parallel_processing = None
    if parallel_type:
        parallel_processing = (parallel_type, num_processors)
        
    #Create a tqdm progress bar
    global linear_pbar
    on_generation = None
    if verbose == True:
        linear_pbar = tqdm(total=num_generations, desc="Genetic Algorithm Progress", unit="gen")
        on_generation = linear_on_generation

    # Create a PyGAD instance
    ga_instance = pygad.GA(num_generations=num_generations,
                        num_parents_mating=num_individuals//2,
                        initial_population=initial_population,
                        fitness_func=linear_fitness_function,
                        mutation_percent_genes=mutation_rate, 
                        parallel_processing=parallel_processing,
                        suppress_warnings=True,
                        #on_generation=on_generation
                        )

    # Run the Genetic Algorithm
    ga_instance.run()

    #ga_instance.plot_fitness()
    
    # Get the fitness value in each generation
    fitness_values = ga_instance.best_solutions_fitness

    # Get the best solution
    best_solution = ga_instance.best_solution()

    # Access the best fitness value and corresponding coefficients
    best_fitness = best_solution[1]
    best_coefficients = best_solution[0]

    return fitness_values, best_fitness, best_coefficients


#__________________________  Use MLP as a Model __________________________   
# Define the fitness function
def mlp_fitness_function(ga_instance, solution, solution_idx):
    nn.set_weights(solution)
    predictions = nn.forward(data)
    
    # Normalize predictions
    min_value = np.min(predictions)
    max_value = np.max(predictions)


    if min_value == max_value:
        # If min_value equals max_value, avoid normalization
        # Set all predictions to a fixed value
        print("***************************************")
        print(solution, end="\n")
        print("***************************************")
        
        predictions.fill(min_value)

    else:
        # Normalize new_indicator to range [0, 100]
        predictions = ((predictions - min_value) / (max_value - min_value)) * 100

    INITIAL_BALANCE = 10000
    cash_balance = INITIAL_BALANCE
    bitcoin_amount = 0
    total_balance = cash_balance
    profit = 0
    
    for i in range(len(prices)):
        indicator_value = predictions[i]
        price = prices[i]
        
        total_balance =  cash_balance + (bitcoin_amount * price)
        bitcoin_amount = ((indicator_value/100) * total_balance) / price
        cash_balance = total_balance - (bitcoin_amount * price)
        profit = total_balance - INITIAL_BALANCE

    return profit

def mlp_on_generation(ga_instance):
    #global pbar
    mlp_pbar.update(1)

def mlp_genetic_algorithm(data_param, prices_param, num_individuals, num_genes, num_generations, mutation_rate, initial_population, model, parallel_type, num_processors, verbose=None):
    global data, prices, nn
    
    # The neural network architecture
    nn = model
    
    # Create the initial population
    num_individuals = num_individuals
    num_genes = num_genes
    initial_population = initial_population

    # Configure Genetic Algorithm parameters
    num_generations = num_generations
    mutation_rate = mutation_rate

    # Data
    data = data_param
    prices = prices_param
    # Parallel Processing Setup
    parallel_processing = None
    if parallel_type:
        parallel_processing = (parallel_type, num_processors)

    # Create a tqdm progress bar
    global mlp_pbar
    on_generation = None
    if verbose == True:
        mlp_pbar = tqdm(total=num_generations, desc="Genetic Algorithm Progress", unit="gen")
        on_generation = mlp_on_generation


    # Create a PyGAD instance
    ga_instance = pygad.GA(num_generations=num_generations,
                        num_parents_mating=num_individuals//2,
                        initial_population=initial_population,
                        fitness_func=mlp_fitness_function,
                        mutation_percent_genes=mutation_rate, 
                        parallel_processing=parallel_processing,
                        suppress_warnings=True,
                        on_generation=on_generation)
    
    # Run the Genetic Algorithm
    ga_instance.run()

    #ga_instance.plot_fitness()
    
    # Get the fitness value in each generation
    fitness_values = ga_instance.best_solutions_fitness

    # Get the best solution
    best_solution = ga_instance.best_solution()

    # Access the best fitness value and corresponding coefficients
    best_fitness = best_solution[1]
    best_coefficients = best_solution[0]

    return fitness_values, best_fitness, best_coefficients