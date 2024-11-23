import numpy as np


class LogisticRegression():

    def __init__(self, n_feature = 13, learning_rate = 1e-5, epochs = 100, tolerance = None, patience = 10, threshold = 0.5):
        self.lr = learning_rate
        self.epochs = epochs
        # self.W = np.random.random(n_feature + 1)
        self.W = np.random.uniform(-0.01, 0.01, n_feature + 1)
        self.loss = []
        self.best_loss = np.inf
        self.tol = tolerance
        self.patience = patience
        self.threshold = threshold
        
    def _linear_tf(self, X):
        return X @ self.W
    
    def _sigmoid(self, z):
        out = 1. / (1. + np.exp(-z))
        return out

    def _map_y(self, y):
        mapper = lambda y: 0 if y == 1 else 1
        return np.array([mapper(yi) for yi in y])

    def _predict_probability(self, X):
        z = self._linear_tf(X)
        return self._sigmoid(z)

    def _cross_entropy_loss(self, y, y_pred):
        epsilon = 1e-8
        loss = -np.mean(y * np.log(y_pred + epsilon) + (1-y) * np.log(1-y_pred + epsilon))
        return loss
    
    def _gradient(self, X, y, y_pred):
        # if-else for single sample and multiple samples. This is a pretty naive solution but it works in our case.
        if isinstance(y, np.ndarray):
            return -(y - y_pred) @ X / y.size
        else:
            return -(y - y_pred) * X / y.size

    
    def _preprocess_data(self, X):
        m, n = X.shape
        X_ = np.empty([m, n+1])
        X_[:, 0] = 1
        X_[:, 1:] = X

        return X_

    def BGD(self, X_train, y):

        X_train_bar = self._preprocess_data(X_train)
        y = self._map_y(y)
        epoch_no_improve = 0

        for epoch in range(self.epochs):
            shuffle_index = np.random.permutation(X_train_bar.shape[0])
            X_train_bar = X_train_bar[shuffle_index]
            y = y[shuffle_index]

            y_pred = self._predict_probability(X_train_bar)

            # why is loss here valid? 
            # loss generally should be a scalar. In our previous example, the loss was calculated for each data feature, but here in the logistic func, it is different.
            loss = self._cross_entropy_loss(y, y_pred)
            self.loss.append(loss)

            gradient = self._gradient(X_train_bar, y, y_pred)
            self.W -= self.lr * gradient

            if self.tol is not None:
                if loss < self.best_loss - self.tol and loss != 0:
                    # the case where the new loss is good enough
                    # i.e. change of loss is bigger than the tolerance
                    self.best_loss = loss # we update the best loss
                    epoch_no_improve = 0
                elif np.abs(loss - self.best_loss) < self.tol:
                    epoch_no_improve += 1
                    if epoch_no_improve == self.patience:
                        print(f'Early stopping at epoch {epoch}')
                        return        


    def SGD(self, X_train, y):
        X_train_bar = self._preprocess_data(X_train)
        y = self._map_y(y)
        epoch_no_improve = 0

        for epoch in range(self.epochs):
            shuffle_index = np.random.permutation(X_train_bar.shape[0])
            X_train_bar = X_train_bar[shuffle_index]
            y = y[shuffle_index]
            # we have this part unchanged
            
            for i in range(X_train_bar.shape[0]):
                x_bar = X_train_bar[i]
                y_pred = self._predict_probability(x_bar)
                loss = self._cross_entropy_loss(y, y_pred)
                self.loss.append(loss)
                # x_bar is one sample here
                gradient = self._gradient(x_bar, y[i], y_pred)
                self.W -= self.lr * gradient

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

    def MBGD(self, X_train, y, batch_size = 16):
        X_train_bar = self._preprocess_data(X_train)
        y = self._map_y(y)
        epoch_no_improve = 0

        for epoch in range(self.epochs):
            shuffle_index = np.random.permutation(X_train_bar.shape[0])
            X_train_bar = X_train_bar[shuffle_index]
            y = y[shuffle_index]
            # we have this part unchanged
            
            for i in range(0, X_train_bar.shape[0], batch_size): 
                # Here, Python's slicing mechanism will automatically handle out-of-bounds cases. 
                # If the size is not divisible, the size of the last batch will be smaller than batch_size.
                x_bar = X_train_bar[i:i+batch_size]
                y_slice = y[i:i+batch_size]
                y_pred = self._predict_probability(x_bar)
                loss = self._cross_entropy_loss(y_slice, y_pred)
                self.loss.append(loss)
                # x_bar is one sample here
                gradient = self._gradient(x_bar, y[i:i+batch_size], y_pred)
                self.W -= self.lr * gradient

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
    
    def predict(self, X):
        X = self._preprocess_data(X)
        y_pred = self._predict_probability(X)
        return np.where(y_pred >= self.threshold, 1, 0)
    
    def batch_cross_entropy_loss(self, y, y_pred):
        loss = np.where( y == y_pred, 0, 1)
        return loss



    