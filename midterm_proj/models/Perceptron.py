from matplotlib import pyplot as plt
import numpy as np


class Perceptron():

    def __init__(self, n_feature = 13, learning_rate = 1e-3, epochs = 100, tolerance = None, patience = 10):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.W = np.random.random(n_feature + 1) * 0.5
        self.W = np.random.uniform(0.01, 0.01, n_feature + 1)
        self.loss = []
        self.best_loss = np.inf

        self.tol = tolerance
        self.patience = patience
        
    
    def _loss(self, y, y_pred):
        return - y_pred * y if y_pred * y < 0 else 0

    def _gradient(self, x_bar, y, y_pred):
        return -y * x_bar if y_pred * y < 0 else 0

    def batch_loss(self, y, y_pred):
        loss = np.where( y == y_pred, 0, np.abs(y * y_pred))
        return loss

    def batch_gradient(self, x_bar, y, y_pred):
        gradient = np.where((y_pred * y)[:, np.newaxis] < 0, -y[:,np.newaxis] * x_bar, 0)
        return np.sum(gradient, axis=0)

    def _preprocess_data(self, X):
        m, n = X.shape
        X_ = np.empty([m, n+1])
        X_[:, 0] = 1
        X_[:, 1:] = X
        return X_

    def _map_y(self, y):
        mapper = lambda y: -1 if y == 1 else 1
        return np.array([mapper(yi) for yi in y])
    
    def _predict(self, X):
        return X @ self.W

    def SGD(self, X_train, y):
        X_train_bar = self._preprocess_data(X_train)
        # breakout = False
        y = self._map_y(y)
        epoch_no_improve = 0
        # self.loss.append(self._loss(y, self._predict(X_train_bar)))

        for epoch in range(self.epochs):
            shuffle_index = np.random.permutation(X_train_bar.shape[0])
            X_train_bar = X_train_bar[shuffle_index]
            y = y[shuffle_index]
            
            for i in range(X_train_bar.shape[0]):
                x_bar = X_train_bar[i]
                y_pred = self._predict(x_bar)
                loss = self._loss(y[i], y_pred)
                self.loss.append(loss)

                # A simple grad desc without considering earlystopping
                grad = self._gradient(x_bar, y[i], y_pred)
                self.W -= self.learning_rate * grad

                # ----------------------------
                #      end of grad desc      
                #         one sample
                # ----------------------------

                # update-based early stopping
                if self.tol is not None:
                    if loss < self.best_loss - self.tol and loss != 0:
                        self.best_loss = loss
                        epoch_no_improve = 0
                    elif np.abs(loss - self.best_loss) < self.tol:
                        epoch_no_improve += 1
                        if epoch_no_improve == self.patience:
                            print(f'Early stopping at epoch {epoch}')
                            return

                # Why use another variable called break? Let's first try using return.

    def BGD(self, X_train, y):
        X_train_bar = self._preprocess_data(X_train)
        y = self._map_y(y)
        epoch_no_improve = 0


        for epoch in range(self.epochs):
            shuffle_index = np.random.permutation(X_train_bar.shape[0])
            X_train_bar = X_train_bar[shuffle_index]
            y = y[shuffle_index]

            y_pred = self._predict(X_train_bar)
            loss = self.batch_loss(y, y_pred)
            scalar_loss = np.sum(loss) / X_train_bar.shape[0]
            self.loss.append(scalar_loss) # we sum the loss of all samples, into a scalar

            grad = self.batch_gradient(X_train_bar, y, y_pred)
            self.W -= self.learning_rate * grad

            if self.tol is not None:
                if scalar_loss < self.best_loss - self.tol and scalar_loss != 0:
                    self.best_loss = scalar_loss
                    epoch_no_improve = 0
                elif np.abs(scalar_loss - self.best_loss) < self.tol:
                    epoch_no_improve += 1
                    if epoch_no_improve == self.patience:
                        print(f'Early stopping at epoch {epoch}')
                        return

                

    def plot_loss(self):
        plt.plot(self.loss)
        plt.grid()
        plt.show()  

    def predict(self, X):
        X_bar = self._preprocess_data(X)
        return np.sign(self._predict(X_bar))


        

