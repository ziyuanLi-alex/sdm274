import numpy as np
from tqdm import tqdm

class Module:
    """神经网络基础组件的基类"""
    def __init__(self):
        self.parameters = []
        self.output_cache = None
        self.input_cache = None
        self.isInference = False

    def forward(self, X):
        raise NotImplementedError
        
    def backward(self, dA):
        raise NotImplementedError
    
class Linear(Module):
    """线性层: y = Wx + b"""
    def __init__(self, input_size, output_size):
        super().__init__()
        self.W = np.random.randn(input_size, output_size) * 0.01
        self.b = np.zeros((1, output_size))
        self.parameters = {"W": self.W, "b": self.b}
    
    def forward(self, X):
        if self.isInference:
            return np.dot(X, self.W) + self.b
        self.input_cache = X
        return np.dot(X, self.W) + self.b
    
    def backward(self, dZ):
        X = self.input_cache
        m = X.shape[0]
        self.parameters['dW'] = (dZ.T @ X).T / m
        self.parameters['db'] = np.sum(dZ, axis=0, keepdims=True) / m
        return np.dot(dZ, self.W.T)
    

class Activation(Module):
    """激活函数的基类"""
    def __init__(self):
        super().__init__()
    
    def forward(self, X):
        raise NotImplementedError
    
    def backward(self, dA):
        raise NotImplementedError
    
class Sigmoid(Activation):
    def forward(self, X):
        A = 1 / (1 + np.exp(-X))
        if not self.isInference:
            self.output_cache = A
        return A
    
    def backward(self, dA):
        A = self.output_cache
        return dA * A * (1 - A)

class ReLU(Activation):
    def forward(self, X):
        if not self.isInference:
            self.output_cache = X
        return np.maximum(0, X)
    
    def backward(self, dA):
        return dA * (self.output_cache > 0)
    

class MLP(Module):
    def __init__(self, layers, epochs=1000, lr=0.01, input_shape=1, output_shape=1):
        super().__init__()
        
        self.layers = layers
        self.epochs = epochs
        self.lr = lr

        self.loss = []
        self.validation_loss = []

        self.input_shape = input_shape
        self.output_shape = output_shape

    def forward(self, X):
        # Also used as the function to get the output of the model
        A = X  # input layer, caches the input
        for layer in self.layers:
            A = layer.forward(A)
        return A

    def backward(self, dA):
        dZ = dA 
        # again we externally call the last layer
        for layer in reversed(self.layers):
            dZ = layer.backward(dZ)

    def inference(self, X):
        X = X.reshape(-1, self.input_shape)

        for layer in self.layers:
            layer.isInference = True
        
        A = X
        for layer in self.layers:
            A = layer.forward(A)
        
        for layer in self.layers:
            layer.isInference = False
        
        return A
        
    def get_loss(self, A, Y):
        pass

    def get_loss_grad(self, A, Y): # predicted first, true second
        pass
    
    def update_params(self, lr):
        for layer in self.layers:
            if isinstance(layer, Linear):
                layer.W -= lr * layer.parameters['dW']
                layer.b -= lr * layer.parameters['db']

    def predict(self, X):
        pass
    
    def reset(self):
        for layer in self.layers:
            if isinstance(layer, Linear):
                layer.reset()
        self.loss = []
                
    def train_BGD(self, X, Y, epochs=None, lr=None):
        if epochs is None:
            epochs = self.epochs
        if lr is None:
            lr = self.lr

        # expands X to (m, features) 
        # feature number here is coded in self.input_shape
        X = X.reshape(-1, self.input_shape)

        for epoch in tqdm(range(epochs), desc="Training BGD"):
            indices = np.random.permutation(X.shape[0])
            X_shuffled = X[indices]
            Y_shuffled = Y[indices]
            
            # forward pass
            # it iterates all the layers except the last one
            A = self.forward(X_shuffled) 

            # here A is the predicted array
            # compute the loss using the MSE layer
            loss = self.get_loss(A, Y_shuffled)
            self.loss.append(loss)

            # backward pass using the MSE layer
            dA = self.get_loss_grad(A, Y_shuffled)
            self.backward(dA)
            
            # update the parameters
            self.update_params(lr)

    def train_MBGD(self, X, Y, batch_size=32, epochs=None, lr=None):
        if epochs is None:
            epochs = self.epochs
        if lr is None:
            lr = self.lr

        X = X.reshape(-1, self.input_shape)
        m = X.shape[0]

        for epoch in tqdm(range(epochs), desc="Training MBGD"):
            indices = np.random.permutation(m)
            X_shuffled = X[indices]
            Y_shuffled = Y[indices]

            for i in range(0, m, batch_size):
                X_batch = X_shuffled[i:i + batch_size]
                Y_batch = Y_shuffled[i:i + batch_size]

                Y_batch = Y_batch.reshape(-1, 1)
                X_batch = X_batch.reshape(-1, self.input_shape)

                A = self.forward(X_batch)
                loss = self.get_loss(A, Y_batch)
                self.loss.append(loss)

                # A here is the predicted array
                dA = self.get_loss_grad(A, Y_batch)
                self.backward(dA)
                self.update_params(lr)
                # Validation on the test dataset
                # A_test = self.inference(X_test)
                # val_loss = self.get_loss(A_test, y_test)
                # self.validation_loss.append(val_loss)

    def train_SGD(self, X, Y, epochs=None, lr=None):
        if epochs is None:
            epochs = self.epochs
        if lr is None:
            lr = self.lr

        X = X.reshape(-1, self.input_shape)
        m = X.shape[0]

        for epoch in tqdm(range(epochs), desc="Training SGD"):
            indices = np.random.permutation(m)
            X_shuffled = X[indices]
            Y_shuffled = Y[indices]

            for i in range(m):
                X_sample = X_shuffled[i:i + 1]
                Y_sample = Y_shuffled[i:i + 1]

                A = self.forward(X_sample)
                loss = self.get_loss(A, Y_sample)
                self.loss.append(loss)

                dA = self.get_loss_grad(A, Y_sample)
                self.backward(dA)
                self.update_params(lr)
    
    def train_k_fold(self, X, Y, k=5, epochs=None, lr=None):
        if epochs is None:
            epochs = self.epochs
        if lr is None:
            lr = self.lr

        m = X.shape[0]
        fold_size = m // k # integer division, i.e. 103 // 5 = 20
        indices = np.arange(m) # 0, 1, 2, ..., m-1
        np.random.shuffle(indices) # shuffle the indices, train randomly

        fold = 1
        for i in range(k): # iterate over the folds, totally we train for k times
            val_start = i * fold_size
            val_end = val_start + fold_size if i < k - 1 else m # if it's the last fold, use the remaining data

            val_indices = indices[val_start:val_end] # takes the validation indices
            train_indices = np.concatenate([indices[:val_start], indices[val_end:]]) # takes all the indices except the validation indices

            X_train = X[train_indices]
            Y_train = Y[train_indices]
            X_val = X[val_indices]
            Y_val = Y[val_indices]

            self.reset()
            self.train_MBGD(X_train, Y_train, epochs=epochs, lr=lr) # we still use the mini-batch gradient descent, with a default bs of 32.
            
            A_val = self.inference(X_val)
            loss = self.get_loss(A_val, Y_val)
            self.validation_loss.append(loss)

            print(f'Fold {fold}, Loss: {loss}')
            fold += 1

class MLPClassifier(MLP):

    def __init__(self, layers, epochs=1000, lr=0.01, input_shape=1, output_shape=1):
        super().__init__(layers, epochs, lr, input_shape, output_shape)
        self.threshold = 0.5
        
    def get_loss(self, y_true, y_pred):
        y_pred = y_pred.reshape(-1, 1)
        # Clip predictions to prevent log(0)
        y_pred = np.clip(y_pred, 1e-12, 1. - 1e-12)
        # Compute cross-entropy loss
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def get_loss_grad(self, y_pred, y_true):
        # The loss here is the cross-entropy loss
        # It accepts matrix with the shape (x, n_classes)
        # Where x is the number of samples and n_classes is the number of classes
        
        y_pred = y_pred.reshape(-1, 1)
        # Clip predictions to prevent division by zero
        y_pred = np.clip(y_pred, 1e-12, 1. - 1e-12)
        # Compute gradient of cross-entropy loss
        return (y_pred - y_true) / (y_pred * (1 - y_pred))
    
    def predict(self, X):
        # Forward pass
        y_pred = self.forward(X)
        
        # Convert the output to binary
        return (y_pred > self.threshold).astype(int)
    

