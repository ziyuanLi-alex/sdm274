import numpy as np

class Module:

    def __init__(self):
        self.parameters = []

    def forward(self):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError

    def parameters(self):
        return self.parameters


class Linear(Module):
    
    def __init__(self, input_size, output_size):
        super().__init__()
        self.input_size = input_size # input features
        self.output_size = output_size # output features
        self.W = np.random.randn(input_size, output_size) * 0.1 
        self.b = np.zeros((1, output_size))
        self.parameters = {"W": self.W, "b": self.b}
        self.input_cache = None


    def forward(self, X):
        self.input_cache = X
        return np.dot(X, self.W) + self.b # X(m, input_size) @ W(input_size, output_size) + b(1, output_size) = Y(m, output_size)
        # X @ W + b = Y

    def backward(self, dZ):
        # Retrieve the cached input
        X = self.input_cache
        m = X.shape[0] # means samples
        
        # Calculate gradients with respect to weights, bias, and input
        dW = (dZ.T @ X) / m # dZ.T ( output_size, m) @ X (m, input_size) = dW (output_size, input_size)
        dW = dW.T  # Transpose to match the shape of W (input_size, output_size)
        db = np.sum(dZ, axis=0, keepdims=True) / m # db (1, output_size)
        dX = np.dot(dZ, self.W.T) # dZ (m, output_size) @ W.T (output_size, input_size) = dX (m, input_size)
        
        # Save gradients for parameter updates
        self.parameters['dW'] = dW
        self.parameters['db'] = db
        
        return dX

