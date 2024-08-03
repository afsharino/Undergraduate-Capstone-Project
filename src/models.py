# Import Libraries
import numpy as np

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