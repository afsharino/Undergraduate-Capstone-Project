# Import libraries
import numpy as np
import pygad
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
#from tensorflow.python.keras import layers

# use linear model 
def linear_genetic_algorithm(data, prices, num_individuals, num_genes, num_generations, mutation_rate, initial_population):
    # Define the fitness function
    def fitness_function(ga_instance, solution, solution_idx):
        new_indicator = np.dot(data, solution)
        min_value = np.min(new_indicator)
        max_value = np.max(new_indicator)
        # Normalize new_indicator to range [0, 100]
        new_indicator = ((new_indicator - min_value) / (max_value - min_value)) * 100
        
        INITIAL_BALANCE = 10000
        cash_balance = INITIAL_BALANCE
        bitcoin_amount = 0
        total_balance = cash_balance
        profit = 0
        
        for i in range(len(prices)):
            indicator_value = new_indicator[i]
            price = prices[i]
            
            total_balance =  cash_balance + (bitcoin_amount*price)
            
            bitcoin_amount = ((indicator_value/100) * total_balance) / price
            
            cash_balance = total_balance - (bitcoin_amount*price)
            
            profit = total_balance - INITIAL_BALANCE
        return profit

    # Create the initial population
    num_individuals = num_individuals
    num_genes = num_genes
    initial_population = initial_population

    # Configure Genetic Algorithm parameters
    num_generations = num_generations
    mutation_rate = mutation_rate

    # Create a PyGAD instance
    ga_instance = pygad.GA(num_generations=num_generations,
                        num_parents_mating=num_individuals//2,
                        initial_population=initial_population,
                        fitness_func=fitness_function,
                        mutation_percent_genes=mutation_rate, 
                        parallel_processing=["thread", 50],
                        suppress_warnings=True)

    # Run the Genetic Algorithm
    ga_instance.run()

    # Get the fitness value in each generation
    fitness_values = ga_instance.best_solutions_fitness

    # Get the best solution
    best_solution = ga_instance.best_solution()

    # Access the best fitness value and corresponding coefficients
    best_fitness = best_solution[1]
    best_coefficients = best_solution[0]

    return fitness_values, best_fitness, best_coefficients


#__________________________  Use MLP as a Model __________________________
mlp_data = None
mlp_prices = None
mlp_input_layer = None
mlp_hidden_layer_1 = None      
mlp_bias_1 = None
mlp_output_layer = None
mlp_model = None

#__________________________ mlp_fitness_function __________________________

def mlp_fitness_function(ga_instance, solution, solution_idx):
    try:
        
        reshaped_solution = solution.reshape((-1,))

        weights = reshaped_solution[:mlp_input_layer * mlp_hidden_layer_1].reshape((mlp_input_layer, mlp_hidden_layer_1))
        biases1 = reshaped_solution[mlp_input_layer * mlp_hidden_layer_1:mlp_input_layer * mlp_hidden_layer_1 + mlp_bias_1]
        weights2 = reshaped_solution[mlp_input_layer * mlp_hidden_layer_1 + mlp_bias_1:mlp_input_layer * mlp_hidden_layer_1 + mlp_bias_1 + mlp_hidden_layer_1 * mlp_output_layer].reshape((mlp_hidden_layer_1, mlp_output_layer))
        biases2 = reshaped_solution[mlp_input_layer * mlp_hidden_layer_1 + mlp_bias_1 + mlp_hidden_layer_1 * mlp_output_layer:]

        mlp_model.set_weights([weights, biases1, weights2, biases2])

        new_indicator = mlp_model.predict(mlp_data).reshape(-1)

        min_value = np.min(new_indicator)
        max_value = np.max(new_indicator)
        new_indicator = ((new_indicator - min_value) / (max_value - min_value)) * 100
        print(new_indicator.shape)
        INITIAL_BALANCE = 10000
        cash_balance = INITIAL_BALANCE
        bitcoin_amount = 0
        total_balance = cash_balance
        profit = 0

        for i in range(len(mlp_prices)):
            indicator_value = new_indicator[i]
            price = mlp_prices[i]

            total_balance = cash_balance + (bitcoin_amount * price)
            bitcoin_amount = ((indicator_value / 100) * total_balance) / price
            cash_balance = total_balance - (bitcoin_amount * price)
            profit = total_balance - INITIAL_BALANCE

        return profit
    except Exception as e:
        print(f"Error in fitness function: {e}")
        raise
    



#__________________________ mlp_genetic_algorithm __________________________
def mlp_genetic_algorithm(data, prices, num_individuals, num_genes, num_generations, mutation_rate, initial_population, nn_layers, model):
   
    global mlp_data
    global mlp_prices
    global mlp_input_layer
    global mlp_hidden_layer_1       
    global mlp_bias_1
    global mlp_output_layer
    global mlp_model

    # Neural net layers
    input_layer, hidden_layer_1,bias_1, output_layer, bias_output = nn_layers
    
    mlp_data = data
    mlp_prices = prices
    mlp_input_layer = input_layer
    mlp_hidden_layer_1 = hidden_layer_1       
    mlp_bias_1 = bias_1
    mlp_output_layer = output_layer
    mlp_model = model
    
    # Create the initial population
    num_individuals = num_individuals
    num_genes = num_genes
    initial_population = initial_population

    # Configure Genetic Algorithm parameters
    num_generations = num_generations
    mutation_rate = mutation_rate

    # Create a PyGAD instance
    ga_instance = pygad.GA(num_generations=num_generations,
                        num_parents_mating=num_individuals//2,
                        initial_population=initial_population,
                        fitness_func=mlp_fitness_function,
                        mutation_percent_genes=mutation_rate, 
                        parallel_processing=["process", 5],
                        suppress_warnings=True)

    # Run the Genetic Algorithm
    ga_instance.run()

    ga_instance.plot_fitness(plot_type="plot", label="MLP Model", color='purple')
    plt.show()

    # Get the best solution
    best_solution = ga_instance.best_solution()

    # Access the best fitness value and corresponding coefficients
    best_fitness = best_solution[1]
    best_coefficients = best_solution[0]

    return best_fitness, best_coefficients