# Import libraries
import numpy as np
import pygad

def genetic_algorithm(data, prices, num_individuals, num_genes, num_generations, mutation_rate):
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
    initial_population = np.random.rand(num_individuals, num_genes)  # Random initialization

    # Configure Genetic Algorithm parameters
    num_generations = num_generations
    mutation_rate = mutation_rate

    # Create a PyGAD instance
    ga_instance = pygad.GA(num_generations=num_generations,
                        num_parents_mating=num_individuals//2,
                        initial_population=initial_population,
                        fitness_func=fitness_function,
                        mutation_percent_genes=mutation_rate, 
                        parallel_processing=["thread", 100],
                        suppress_warnings=True)

    # Run the Genetic Algorithm
    ga_instance.run()

    # Get the best solution
    best_solution = ga_instance.best_solution()

    # Access the best fitness value and corresponding coefficients
    best_fitness = best_solution[1]
    best_coefficients = best_solution[0]

    return best_fitness, best_coefficients

   