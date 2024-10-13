import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Perception:
    def __init__(self,n_feature=1,n_iter=200,lr=0.001,tol=None):
        self.n_feature = n_feature
        self.n_iter = n_iter
        self.lr = lr
        self.tol = tol
        self.W = np.random.random(n_feature+1)*0.5
        self.loss = []
        self.best_loss = np.inf
        self.patience = 20
    
    def _loss(self,y,y_pred):
        # print(y)
        # print(y_pred)
        return -y_pred*y if y_pred*y < 0 else 0

    def _gradient(self,x_bar,y,y_pred):
        return -y*x_bar if y_pred*y <= 0 else 0

    def _preprocess_data(self, X):
        m,n = X.shape
        X_ = np.empty([m,n+1])
        X_[:,0] = 1
        X_[:,1:] = X
        return X_
    
    def _predict(self,X):
        # print(X)
        # print("self.W is")
        # print(self.W)
        return X @ self.W

    def S_Update(self,X,y):
        break_out = False
        epoch_without_improve = 0

        for iter in range(self.n_iter):
            for i,x in enumerate(X):
                y_pred = self._predict(x)
                loss = self._loss(y[i],y_pred)
                self.loss.append(loss)

                if self.tol is not None:
                    # print('loss: ',loss)
                    # print('self.best_loss: ',self.best_loss)
                    if loss < self.best_loss - self.tol:
                        self.best_loss = loss
                        epoch_without_improve = 0
                    elif np.abs(loss-self.best_loss) < self.tol:
                        epoch_without_improve += 1
                        if epoch_without_improve > self.patience:
                            print('Early stop')
                            break_out = True
                            break
                    else:
                        epoch_without_improve = 0
                
                grad = self._gradient(x,y[i],y_pred)
                self.W = self.W - self.lr*grad
            
            if break_out:
                break_out = False
                break
    
    def train(self,X_train,y_train):
        X_train_bar = self._preprocess_data(X_train)
        # print(X_train_bar)
        self.S_Update(X_train_bar,y_train)
        # print(self.W)

    def plot_show(self):
        plt.plot(self.loss)
        plt.grid()
        plt.show()

data = pd.read_csv('C:\\Users\\sunyy\\Desktop\\SUSTECH\\Y\\code\\人工智能与机器学习\\assignment2\\wine.data',header=None)
data = data.drop(data.index[-48:]).sample(frac=0.7).reset_index(drop=True)
print(data)



case1 = data.iloc[:59]
case2 = data.iloc[60:]

X_train = data.iloc[:,1:]
y_train = data.iloc[:,0]
y_train = y_train*2-3

# print("X_train is ")
# print(X_train)

perception = Perception(13,1000,3e-6,1e-10)
perception.train(X_train,y_train)
perception.plot_show()

