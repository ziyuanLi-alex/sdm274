import numpy as np

class LinearRegression():

    def __init__(self, n_feature = 1, epochs = 100, lr = 1e-5):
        self.n_feature = n_feature
        self.epochs = epochs
        self.W = (np.random.random(n_feature + 1) * 0.05).reshape(-1,1)
        self.loss = []
        self.epoch_loss = []
        self.lr = lr

    def _loss(self, y, y_pred):
        return np.sum((y - y_pred) ** 2) / y.size

    def gradient(self, X, y, y_pred):
        y_pred = y_pred.reshape(-1, 1)
        return  X.T @ (y_pred - y) / y.size

    def _preprocess(self, X):
        m, n = X.shape
        X_ = np.empty([m, n + 1])
        X_[:, 0] = 1
        X_[:, 1:] = X
        return X_

    def _predict(self, X):
        return X @ self.W

    def predict(self, X):
        X_ = self._preprocess(X)
        return self._predict(X_)
    
    def BGD(self, X, y):
        X_ = self._preprocess(X)
        self.loss.append(self._loss(y, self._predict(X_)))
        self.epoch_loss.append(self._loss(y, self._predict(X_)))

        for _ in range(self.epochs):
            y_pred = self._predict(X_)
            y = y.reshape(-1, 1)
            self.W -= self.lr * self.gradient(X_, y, y_pred) # gradient descent
            self.loss.append(self._loss(y, y_pred))
            self.epoch_loss.append(self._loss(y, y_pred))
    
    def SGD(self, X, y):
        X_ = self._preprocess(X)
        self.loss.append(self._loss(y, self._predict(X_)))
        self.epoch_loss.append(self._loss(y, self._predict(X_)))

        for _ in range(self.epochs):
            shuffle_index = np.random.permutation(X_.shape[0])
            X_ = X_[shuffle_index]
            y = y[shuffle_index]
            for i in range(len(y)):
                xi = X_[i].reshape(1, -1)  # Select a single data point and reshape to match input dimension
                yi = y[i].reshape(-1)       # Select the corresponding target value and reshape to match output dimension
                y_pred = self._predict(xi)
                self.W -= self.lr * self.gradient(xi, yi, y_pred).flatten()  # Update weights based on single data point
                self.loss.append(self._loss(yi, y_pred))
            
            y_pred_epoch = self._predict(X_)
            self.epoch_loss.append(self._loss(y, y_pred_epoch))
            

    def MBGD(self, X, y, batch_size=32):
        X_ = self._preprocess(X)
        num_batches = len(y) // batch_size
        self.loss.append(self._loss(y, self._predict(X_)))
        self.epoch_loss.append(self._loss(y, self._predict(X_)))

        for _ in range(self.epochs):
            shuffle_index = np.random.permutation(X_.shape[0])
            X_ = X_[shuffle_index]
            y = y[shuffle_index]
            for batch in range(num_batches):
                start = batch * batch_size
                end = start + batch_size
                xi = X_[start:end]  # Select a batch of data points
                yi = y[start:end]   # Select the corresponding target values
                y_pred = self._predict(xi)
                self.W -= self.lr * self.gradient(xi, yi, y_pred)  # Update weights based on the batch
                self.loss.append(self._loss(yi, y_pred))

            # Compute loss for the entire dataset
            y_pred_epoch = self._predict(X_)
            self.epoch_loss.append(self._loss(y, y_pred_epoch))

    def minmax_norm(self, X):
        return (X - X.min()) / (X.max() - X.min())

    def mean_norm(self, X):
        return (X - X.mean()) / X.std()


def threshold_classifier(y_pred, threshold=0.5):
    return np.where(y_pred >= threshold, 1, 0)