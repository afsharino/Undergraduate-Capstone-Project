# # Import Libraries
import numpy as np

class NeuralNetwork:
    def __init__(self, layer_sizes):
        """
        layer_sizes: List of layer sizes. Example: [input_size, hidden_1_size, ..., hidden_n_size, output_size]
        """
        self.num_layers = len(layer_sizes) - 1  # Total number of layers, excluding input layer
        self.weights = []
        self.biases = []
        
        # Initialize weights and biases for each layer
        for i in range(self.num_layers):
            self.weights.append(np.random.randn(layer_sizes[i], layer_sizes[i+1]))
            self.biases.append(np.random.randn(layer_sizes[i+1]))

    def relu(self, x):
        return np.maximum(0, x)

    def linear(self, x):
        return x
    
    def forward(self, X):
        """
        Forward pass through the network with multiple hidden layers.
        """
        activation = X  # Input layer
        
        # Iterate through each hidden layer
        for i in range(self.num_layers - 1):  # Exclude the final layer (output layer)
            z = np.dot(activation, self.weights[i]) + self.biases[i]
            activation = self.relu(z)
        
        # Output layer (assuming linear activation for output)
        z_output = np.dot(activation, self.weights[-1]) + self.biases[-1]
        output = self.linear(z_output)
        
        return output

    def set_weights(self, weights):
        start = 0
        for i in range(self.num_layers):
            weight_size = self.weights[i].size
            bias_size = self.biases[i].size
            
            # Reshape the flattened weights into matrices and vectors
            self.weights[i] = weights[start:start+weight_size].reshape(self.weights[i].shape)
            start += weight_size
            self.biases[i] = weights[start:start+bias_size]
            start += bias_size

    def get_weights(self):
        # Flatten all weights and biases into a single vector
        all_weights = [w.flatten() for w in self.weights]
        all_biases = [b for b in self.biases]
        return np.concatenate(all_weights + all_biases)

