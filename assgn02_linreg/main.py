import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')


np.random.seed(42)
X_train = np.arange(100).reshape(100,1)
w, b  = 1, 10
y_train = w * X_train + b + np.random.normal(0,5,size=X_train.shape)
y_train = y_train.reshape(-1)

class LinearRegression():
    def __init__(self, epochs = 1000, lr = 0.000001, X = None, y = None):
        self.epochs = epochs
        self.lr = lr
        self.X = X
        self.y = y

        self.SGD_losses = np.zeros(epochs)
        self.BGD_losses = np.zeros(epochs)
        self.MBGD_losses = np.zeros(epochs)

    def _loss(self, y, y_pred):
        return np.sum((y_pred - y) **2 ) / y.size

    def _gradient(self, X, y, y_pred):
        return (y_pred - y) @ X / y.size

    def predict(self, X):
        return X @ self.w + self.b 

    def fit_SGD(self, X, y, plot=False):

        n_samples, n_features = X.shape # n_samples指代样本数量，n_features指代特征数量
        # 在这一个线性回归的情况下，我们的简单模型为 y = Ax+B, 特征数量为1，即X的shape为(n_samples, 1)
        # 当然，这里的实现也可以适用于多特征的情况，即X的shape为(n_samples, n_features），例如 Y = AX1 + BX2 + ... + C

        # 这里为了可读性考虑使用w和b，不使用矩阵形式
        self.w = np.random.rand(X.shape[1]) # w的shape为(n_features, ), 在本情况中w的shape为(1, )
        self.b = np.random.rand(1) # b的shape为(1, )，即一个零次的偏移量

        for epoch in range(self.epochs): # epochs指代整个训练集的迭代次数
            
            # 随机打乱训练集
            shuffle_index = np.random.permutation(n_samples)
            X = X[shuffle_index]
            y = y[shuffle_index]
            epoch_loss = self._loss(y, self.predict(X))
            self.SGD_losses[epoch] = epoch_loss

            for i in range(n_samples): # 遍历整个训练集，每次只取一个样本
                y_pred = self.predict(X[i])
                loss = self._loss(y[i], y_pred)
                # self.SGD_losses[epoch] += loss
                grad = self._gradient(X[i], y[i], y_pred)

                # 向负梯度方向更新参数
                self.w -= self.lr * grad
                self.b -= self.lr * grad

                # if i % 10 == 0:
                #     print(f'Epoch {epoch}, Sample {i}, Loss {loss}')
            
            
        if plot:
            plt.plot(self.SGD_losses)
            plt.show()
                    
    def min_max_normalize(self, X):
        X_min = np.min(X, axis=0)
        X_max = np.max(X, axis=0)
        return (X - X_min) / (X_max - X_min)

    def mean_normalize(self, X):
        X_mean = np.mean(X, axis=0)
        X_min = np.min(X, axis=0)
        X_max = np.max(X, axis=0)
        return (X - X_mean) / (X_max - X_min)


    def fit_BGD(self, X, y, plot=False):
        n_samples, n_features = X.shape
        self.w = np.random.rand(X.shape[1]) 
        self.b = np.random.rand(1) 
        
        for epoch in range(self.epochs):
            epoch_loss = self._loss(y, self.predict(X))
            self.BGD_losses[epoch] = epoch_loss # 每一个epoch开头存储一次loss

            y_pred = self.predict(X)
            loss = self._loss(y, y_pred)
            grad = self._gradient(X, y, y_pred)
            self.w -= self.lr * grad
            self.b -= self.lr * grad
            # if epoch % 10 == 0:
            #     print(f'Epoch {epoch}, Loss {loss}')

        if plot:
            plt.plot(self.BGD_losses)
            plt.show()

    def fit_MBGD(self, X, y, batch_size=10, plot=False):
        n_samples, n_features = X.shape
        self.w = np.random.rand(X.shape[1])
        self.b = np.random.rand(1)

        for epoch in range(self.epochs):
            shuffle_index = np.random.permutation(n_samples)
            X = X[shuffle_index]
            y = y[shuffle_index]

            epoch_loss = self._loss(y, self.predict(X))
            self.MBGD_losses[epoch] = epoch_loss

            for i in range(0, n_samples, batch_size):
                X_batch = X[i:i+batch_size]
                y_batch = y[i:i+batch_size]

                y_pred = self.predict(X_batch)
                loss = self._loss(y_batch, y_pred)

                grad = self._gradient(X_batch, y_batch, y_pred)
                self.w -= self.lr * grad
                self.b -= self.lr * grad
        

        if plot:
            plt.plot(self.MBGD_losses)
            plt.show()



    def plot_Xy(self, X, y, title=None):
        plt.scatter(X, y, label='Data')
        plt.plot(X, w * X + b, color='red', label='Ideal')
        plt.plot(X, self.predict(X), color='green', label='Prediction')
        if title:
            plt.title(title)
        plt.legend()
        plt.show()

    def plot_loss(self):
        # plt.plot(np.log(self.BGD_losses + 1e-10), label='BGD', color='red')
        # plt.plot(np.log(self.SGD_losses + 1e-10), label='SGD', color='green')
        # plt.plot(np.log(self.MBGD_losses + 1e-10), label='MBGD', color='blue')
        # plt.grid()
        # plt.legend()
        # plt.show()

        plt.plot(self.BGD_losses, label='BGD', color='red')
        plt.plot(self.SGD_losses, label='SGD', color='green')
        plt.plot(self.MBGD_losses, label='MBGD', color='blue')
        plt.legend()
        plt.show()
    
        
            
if __name__ == "__main__":
    model = LinearRegression(X = X_train, y = y_train, epochs=300, lr=0.00005)
    # model.plot_Xy(X_train, y_train, title='Before Training')
    # model.fit_MBGD(X_train, y_train, batch_size=10)
    # model.fit_SGD(X_train, y_train)
    # model.fit_BGD(X_train, y_train)
    # model.plot_Xy(X_train, y_train, title='After Training')
    # model.plot_loss()

    # model2 = LinearRegression(X = X_train, y = y_train, epochs=500, lr=0.0005) # 0.0005是一个比较好的值，这个值下典型地发散
    # model2.fit_SGD(X_train, y_train)
    # model2.plot_Xy(X_train, y_train, title='After Training')

    X_norm = model.min_max_normalize(X_train)
    y_norm = model.min_max_normalize(y_train)

    # plt.scatter(X_norm, y_norm, label='Data')
    # plt.plot(X_norm, model.min_max_normalize(w * X_norm + b), color='red', label='Ideal')
    # plt.show()

    model_norm = LinearRegression(X = X_norm, y = y_norm, epochs=1000, lr=0.0001)
    model_norm.fit_MBGD(X_norm, y_train, batch_size=10)
    model_norm.fit_SGD(X_norm, y_norm)
    model_norm.fit_BGD(X_norm, y_norm, plot=False)
    

    model_norm.plot_Xy(X_norm, y_norm, title='After Training')
    model_norm.plot_loss()
        

    


        
    
    

