from modules import *
from tqdm import tqdm

# a general MLP model
# classification code is not implemented here
# regression code is implemented here
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

