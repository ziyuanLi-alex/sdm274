import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class MLP:
    def __init__(self, layerRecord, nIter=200):
        # 这里好像用不了tol和patience提前结束(?
        self.layersRecord = []
        for layer in layerRecord:
            self.layersRecord.append(Layer(layer[0],layer[1],layer[2]))
        self.nIter = nIter
        self.loss = []

    def _MSE(self, fact, pred):
        SE = np.square(fact - pred)
        SUM_SE = np.sum(SE)
        # print(fact.shape)
        print(pred)
        return SUM_SE/fact.size

    def _GRADIENT(self, y, y_pred):
        return ((y_pred - y)*2)/y.size
    
    def _forward(self,X):
        for layer in self.layersRecord:
            X = layer.forward(X)
        return X

    def _backward(self, grad):
        for layer in reversed(self.layersRecord):
            grad = layer.backward(grad)
    
    def B_update(self,X,y):
        # print(X.shape)
        X = X.reshape(-1, self.layersRecord[0].inputNum)
        # print(X.shape)
        for i in range(self.nIter):
            a = self.layersRecord[0].W
            b = self.layersRecord[1].W
            c = self.layersRecord[2].W
            y_pred = self._forward(X)
            loss = self._MSE(y, y_pred)
            self.loss.append(loss)
            grad = self._GRADIENT(y,y_pred)
            self._backward(grad)
            for layer in self.layersRecord:
                layer.W -= layer.lr * layer.gradW
                layer.b -= layer.lr * layer.gradB
            
    def plt_show(self):
        print(self.loss[-1])
        plt.plot(self.loss)
        plt.xlabel('time')
        plt.ylabel('loss')
        plt.grid()
        plt.show()
    
    def predict(self, X, y):
        X = X.reshape(-1, self.layersRecord[0].inputNum)
        y_pred = self._forward(X)
        for i in range(len(X)):
            print(X[i],y[i],y_pred[i])
        plt.scatter(X,y,c='b',marker='o')
        plt.scatter(X,y_pred,c='r',marker='^')
        plt.show()

class Layer:
    def __init__(self, inputNum, outputNum, lr):
        self.inputNum = inputNum
        self.outputNum = outputNum
        self.lr = lr
        self.W = np.random.rand(inputNum,outputNum)*0.5
        self.b = np.zeros(outputNum)
        self.befXL = None
        self.befXA = None
        self.gradW = None
        self.gradB = None

    def forward(self, befX):
        self.befXL = befX
        # print(befX.shape)
        # print(self.W.shape)
        record = befX @ self.W + self.b
        self.befXA = (record > 0).astype(float)
        if self.outputNum != 1:
            return np.maximum(0,record)
        else:
            return record
        
    def backward(self, gradBack):
        tmp = None
        if self.outputNum != 1:
            tmp = gradBack * self.befXA
        else:
            tmp = gradBack

        self.gradW = (self.befXL.T @ tmp)/self.befXL.shape[0]
        self.gradB = sum(tmp)/self.befXL.shape[0]
        
        return tmp @ self.W.T



dataA = pd.read_csv('C:\\Users\\sunyy\\Desktop\\SUSTECH\\Y\\code\\人工智能与机器学习\\assignment4\\data',header=None)
data = dataA.sample(frac=0.7,random_state=15)
test = dataA.drop(data.index)

data = data.reset_index(drop=True)
test = test.reset_index(drop=True)
X_train = data.iloc[:,0]
y_train = data.iloc[:,:1]
# print(X_train)
# print(y_train)
X_test = test.iloc[:,0]
y_test = test.iloc[:,:1]
# print(X_test)
# print(y_test)

layers = [[1,32,10],[32,64,10],[64,1,10]]
mlp = MLP(layers,200)

X_train = X_train.to_numpy()
y_train = y_train.to_numpy()
X_test = X_test.to_numpy()
y_test = y_test.to_numpy()
# print(X_train)
# print(y_train)
# X_train = np.array([4,12,8,1,6,21,2,78,10,65,35,29])
# y_train = np.array([[4],[12],[8],[1],[6],[21],[2],[78],[10],[65],[35],[29]])
mlp.B_update(X_train,y_train)
mlp.plt_show()
mlp.predict(X_test,y_test)
