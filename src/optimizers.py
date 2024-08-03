# Import libraries
import numpy as np
import pygad
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
#from tensorflow.python.keras import layers

#__________________________  Use Linear Combination as a Model __________________________
# Define the fitness function
def linear_fitness_function(ga_instance, solution, solution_idx):
    new_indicator = np.dot(linear_data, solution)
    min_value = np.min(new_indicator)
    max_value = np.max(new_indicator)
    # Normalize new_indicator to range [0, 100]
    new_indicator = ((new_indicator - min_value) / (max_value - min_value)) * 100
    
    INITIAL_BALANCE = 10000
    cash_balance = INITIAL_BALANCE
    bitcoin_amount = 0
    total_balance = cash_balance
    profit = 0
    
    for i in range(len(linear_prices)):
        indicator_value = new_indicator[i]
        price = linear_prices[i]
        
        total_balance =  cash_balance + (bitcoin_amount*price)
        
        bitcoin_amount = ((indicator_value/100) * total_balance) / price
        
        cash_balance = total_balance - (bitcoin_amount*price)
        
        profit = total_balance - INITIAL_BALANCE
    return profit

def linear_genetic_algorithm(data_param, prices_param, num_individuals, num_genes, num_generations, mutation_rate, initial_population):
    global linear_data, linear_prices

    # Create the initial population
    num_individuals = num_individuals
    num_genes = num_genes
    initial_population = initial_population

    # Configure Genetic Algorithm parameters
    num_generations = num_generations
    mutation_rate = mutation_rate

    # Data
    linear_data = data_param
    linear_prices = prices_param

    # Create a PyGAD instance
    ga_instance = pygad.GA(num_generations=num_generations,
                        num_parents_mating=num_individuals//2,
                        initial_population=initial_population,
                        fitness_func=linear_fitness_function,
                        mutation_percent_genes=mutation_rate, 
                        parallel_processing=["process", 5],
                        suppress_warnings=True)

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
class NeuralNetwork:
    def __init__(self, input_layer_size, hidden_layer_size, output_layer_size):
        # Initialize weights and biases
        self.weights_input_hidden = np.random.randn(input_layer_size, hidden_layer_size)
        self.bias_hidden = np.random.randn(hidden_layer_size)
        self.weights_hidden_output = np.random.randn(hidden_layer_size, output_layer_size)
        self.bias_output = np.random.randn(output_layer_size)
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def forward(self, X):
        # Forward pass
        self.hidden_layer_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_layer_output = self.relu(self.hidden_layer_input)
        self.output_layer_input = np.dot(self.hidden_layer_output, self.weights_hidden_output) + self.bias_output
        self.output_layer_output = self.relu(self.output_layer_input)
        return self.output_layer_output

    def set_weights(self, weights):
        input_hidden_end = self.weights_input_hidden.size
        self.weights_input_hidden = weights[:input_hidden_end].reshape(self.weights_input_hidden.shape)
        
        bias_hidden_end = input_hidden_end + self.bias_hidden.size
        self.bias_hidden = weights[input_hidden_end:bias_hidden_end]
        
        hidden_output_end = bias_hidden_end + self.weights_hidden_output.size
        self.weights_hidden_output = weights[bias_hidden_end:hidden_output_end].reshape(self.weights_hidden_output.shape)
        
        self.bias_output = weights[hidden_output_end:]
    
    def get_weights(self):
        return np.concatenate([
            self.weights_input_hidden.flatten(),
            self.bias_hidden,
            self.weights_hidden_output.flatten(),
            self.bias_output
        ])
    
# Define the fitness function
def mlp_fitness_function(ga_instance, solution, solution_idx):
    nn.set_weights(solution)
    predictions = nn.forward(mlp_data)
    
    # Normalize predictions
    predictions = ((predictions - predictions.min()) / (predictions.max() - predictions.min())) * 100
    #predictions = predictions.reshape(-1)
    
    INITIAL_BALANCE = 10000
    cash_balance = INITIAL_BALANCE
    bitcoin_amount = np.zeros(len(mlp_prices))
    total_balance = np.zeros(len(mlp_prices))

    total_balance[0] = INITIAL_BALANCE
    for i in range(1, len(mlp_prices)):
        total_balance[i] = cash_balance + bitcoin_amount[i-1] * mlp_prices[i]
        bitcoin_amount[i] = (predictions[i] / 100) * total_balance[i] / mlp_prices[i]
        cash_balance = total_balance[i] - bitcoin_amount[i] * mlp_prices[i]

    profit = total_balance[-1] - INITIAL_BALANCE
    return profit

def mlp_genetic_algorithm(data_param, prices_param, num_individuals, num_genes, num_generations, mutation_rate, initial_population, model):
    global mlp_data, mlp_prices, nn
    
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
    mlp_data = data_param
    mlp_prices = prices_param

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

    #ga_instance.plot_fitness()
    
    # Get the fitness value in each generation
    fitness_values = ga_instance.best_solutions_fitness

    # Get the best solution
    best_solution = ga_instance.best_solution()

    # Access the best fitness value and corresponding coefficients
    best_fitness = best_solution[1]
    best_coefficients = best_solution[0]

    return fitness_values, best_fitness, best_coefficients