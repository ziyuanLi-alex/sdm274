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
        # self.W = np.array([ 11.68192425,  81.62188265, -19.21031923,   6.6875174,  306.4937617
#   ,71.22147507 ,  2.69035086, -15.80533329,   3.81183298 ,  6.41101451,
#  -44.84376815 , 14.35953541,  11.31809862 ,-18.63047477])
        self.loss = []
        self.best_loss = np.inf
        self.patience = 20
    
    def _loss(self,y,y_pred):
        # print(y)
        # print(y_pred)
        # print('1',y_pred*y)
        return -y_pred*y if y_pred*y < 0 else 0
    
    def _B_loss(self,y,y_pred):
        cnt = 0
        for i in range(len(y_pred)):
            cnt += self._loss(y[i],y_pred[i])
        return cnt/(len(y_pred))

    def _gradient(self,x_bar,y,y_pred):
        # print(y_pred,y)
        # print('gradient',y*x_bar)
        return -y*x_bar if y_pred*y <= 0 else 0
    
    def _B_gradient(self,x_bar,y,y_pred):
        cnt = np.zeros(self.n_feature+1)
        for i in range(len(x_bar)):
            cnt += self._gradient(x_bar[i],y[i],y_pred[i])

        # print('cnt',cnt)
        return cnt

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
    

    def B_Update(self,X,y):
        break_out = False
        epoch_without_improve = 0

        for iter in range(self.n_iter):
            y_pred = self._predict(X)
            loss = self._B_loss(y,y_pred)
            self.loss.append(loss)
            if self.tol is not None:
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
            grad = self._B_gradient(X,y,y_pred)
            
            self.W = self.W - self.lr*grad

            if break_out:
                break_out = False
                break


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
                # print('grad',grad)
                self.W = self.W - self.lr*grad
            
            if break_out:
                break_out = False
                break
    
    def S_train(self,X_train,y_train):
        X_train_bar = self._preprocess_data(X_train)
        print(X_train_bar)
        self.S_Update(X_train_bar,y_train)
        print(self.W)
    def B_train(self,X_train,y_train):
        X_train_bar = self._preprocess_data(X_train)
        print(X_train_bar)
        self.B_Update(X_train_bar,y_train)
        print(self.W)

    def predict(self,X):
        return self._preprocess_data(X) @ self.W

    def plot_show(self):
        plt.plot(self.loss)
        plt.xlabel('time')
        plt.ylabel('loss')
        plt.grid()
        plt.show()

    def plot_show_test(self,X_test,y_test):
        y_predict = self.predict(X_test)
        x = []
        for i in range(1,len(y_test)+1):
            x.append(i)
        y_predict = y_predict/np.abs(y_predict)
        plt.scatter(x,y_predict,c='b',marker='o')
        plt.scatter(x,y_test,c='r',marker='^')
        plt.show()

        result = [0,0,0,0]
        for i in range(len(x)):
            if(y_predict[i]==1 and y_test[i]==1):
                result[0] += 1
            elif(y_predict[i]==1 and y_test[i]==-1):
                result[1] += 1
            elif(y_predict[i]==-1 and y_test[i]==1):
                result[2] += 1
            else:
                result[3] += 1
        return result

dataA = pd.read_csv('C:\\Users\\sunyy\\Desktop\\SUSTECH\\Y\\code\\人工智能与机器学习\\assignment2\\wine.data',header=None)
dataA = dataA.drop(dataA.index[-48:])
data = dataA.sample(frac=0.7,random_state=15)
test = dataA.drop(data.index)

data = data.reset_index(drop=True)
test = test.reset_index(drop=True)

X_train = data.iloc[:,1:]
y_train = data.iloc[:,0]
y_train = y_train*2-3

X_test = test.iloc[:,1:]
y_test = test.iloc[:,0]
y_test = y_test*2-3



# print("X_train is ")
# print(X_train)


perceptionB = Perception(13,500,1e-7,1e-9)
perceptionB.B_train(X_train,y_train)
perceptionB.plot_show()

result = perceptionB.plot_show_test(X_test,y_test)

print(result)
accuracy = (result[0]+result[3])/(result[0]+result[1]+result[2]+result[3])
recall = (result[0])/(result[0]+result[2])
precision = (result[0])/(result[0]+result[1])
F1 = (2*precision*recall)/(precision+recall)

print('Batch Update')
print('Accuracy =',  accuracy)
print('Recall =', recall)
print('Precision =', precision)
print('F1 score =', F1)

perceptionS = Perception(13,1000,3e-6,1e-10)
perceptionS.S_train(X_train,y_train)
perceptionS.plot_show()

result = perceptionS.plot_show_test(X_test,y_test)

print(result)
accuracy = (result[0]+result[3])/(result[0]+result[1]+result[2]+result[3])
recall = (result[0])/(result[0]+result[2])
precision = (result[0])/(result[0]+result[1])
F1 = (2*precision*recall)/(precision+recall)

print('Stochastic Update')
print('Accuracy =',  accuracy)
print('Recall =', recall)
print('Precision =', precision)
print('F1 score =', F1)
