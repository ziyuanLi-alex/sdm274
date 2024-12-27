import sys
import os
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.preprocessing import load_data
from models.Neural import *
from utils.random_seed import set_seed
from utils.evaluation import Evaluation
import numpy as np

import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from models.SVM import OVOSVM
from models.Adaboost import AdaBoost

set_seed(42)
eval = Evaluation()

def MLP_analysis():

    layers = [
        Linear(input_size=7, output_size=16),
        ReLU(),
        Linear(input_size=16, output_size=16),
        ReLU(),
        Linear(input_size=16, output_size=2),
        Sigmoid()
    ]

    epochs = 3000
    lr = 5e-2

    mlp = MLPMultiClass(layers, epochs=epochs, lr=lr, input_shape=X_train.shape[1], output_shape=2)
    mlp.train_MBGD(X_train_std, y_train_onehot)

    loss_history = mlp.loss
    length = len(loss_history)

    # Plot the loss history
    plt.figure(figsize=(10, 6))
    plt.plot(range(length), loss_history, label='Training Loss')
    plt.xlabel('Batches')
    plt.ylabel('Loss')
    plt.title('Training Loss over Batches')
    plt.legend()
    plt.grid(True)
    plt.show()

    y_pred = mlp.predict(X_test_std)

    y_test_numerical = np.argmax(y_test_onehot, axis=1)

    from sklearn.metrics import confusion_matrix
    import seaborn as sns

    cf_matrix = confusion_matrix(y_test_numerical, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

def SVM_analysis():
    classifiers = {
        'Linear SVM': OVOSVM(C=1.0, kernel='linear'),
        'RBF SVM': OVOSVM(C=1.0, kernel='rbf', gamma=0.1)
    }

    eval = Evaluation()

    for name, clf in classifiers.items():
        print(f"\nTraining {name}...")
        
        # 训练模型
        clf.fit(X_train, y_train)
        
        # 在训练集上评估
        train_pred = clf.predict(X_train)
        # train_acc = accuracy_score(y_train, train_pred)
        train_acc = eval.calculate_accuracy(y_train, train_pred)
        print(f"{name} Training Accuracy: {train_acc:.4f}")
        
        # 在测试集上评估
        test_pred = clf.predict(X_test)
        # test_acc = accuracy_score(y_test, test_pred)
        test_acc = eval.calculate_accuracy(y_test, test_pred)
        print(f"{name} Test Accuracy: {test_acc:.4f}")
        
        # 输出详细的分类报告
        print(f"\n{name} Classification Report:")
        print(classification_report(y_test, test_pred))

def Adaboost_analysis():
    ada = AdaBoost(n_estimators=50)
    ada.fit(X_train, y_train_ada)
    y_pred_ada = ada.predict(X_test)
    accuracy = eval.calculate_accuracy(y_test_ada, y_pred_ada)
    print(f"测试集准确率: {accuracy:.4f}")



if __name__ == "__main__":
    # 加载数据
    X, y = load_data()
    X = X.values if hasattr(X, 'values') else X

    # 移除标签为2的数据
    mask = y != 2
    X = X[mask]
    y = y[mask]
    y_ada = np.where(y == 1, 1, -1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train_ada, X_test_ada, y_train_ada, y_test_ada = train_test_split(X, y_ada, test_size=0.2, random_state=42, stratify=y)

    sc = StandardScaler()
    X_train_std = sc.fit_transform(X_train)
    X_test_std = sc.transform(X_test)

    y_test_onehot = pd.get_dummies(y_test).values.astype(np.float64)
    y_train_onehot = pd.get_dummies(y_train).values.astype(np.float64)
    y_train_ada = np.where(y_train == 1, 1, -1)
    y_test_ada = np.where(y_test == 1, 1, -1)


    # MLP_analysis()

    # SVM_analysis()

    Adaboost_analysis()





    









    