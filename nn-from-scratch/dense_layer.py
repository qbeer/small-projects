import numpy as np

class DenseLayer:
    def __init__(self, input_size, layer_size, activation_fn):
        self.input_size = input_size
        scaler = np.sqrt(2./(self.input_size + layer_size))
        self.weights = np.random.normal(size=(layer_size, input_size)) * scaler # e.g. 20 * 784
        self.biases = np.random.normal(size=layer_size) * scaler
        self.activation_fn = activation_fn
    def forward_pass(self, input_value):
        """
            Multiplying by W and adding b then applying the activation.
        """
        self.input = input_value
        self.activation_input = np.matmul(self.input, self.weights.T) + self.biases # (20 * 784) * (2, 784)T + (20, 2)
        self.output = self.activation_fn.call(self.activation_input)
        return self.output
    def backprop(self, previous_gradient):
        """
            Calculating the gradient by the input df/dx = f'(Wx+b) * W 
        """
        self.activation_derivative = self.activation_fn.derivative(self.activation_input) # f'(Wx + b)
        self.intermediate_gradient = np.multiply(previous_gradient, self.activation_derivative) # f'(Wx + b) * received_grad -> recursive gradient flow
        self.bias_derivative = self.intermediate_gradient # df/db = f'(Wx + b) * 1
        self.weights_derivative = np.outer(self.intermediate_gradient, self.input) # f'(Wx + b) * x
        self.gradient = np.matmul(self.intermediate_gradient, self.weights)
        return self.gradient
    def apply_gradients(self, lr):
        self.weights -= lr*self.weights_derivative
        self.biases -= lr*self.bias_derivative