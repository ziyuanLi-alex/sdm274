import sys
import os
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.preprocessing import load_data
from models.Neural import *
from utils.random_seed import set_seed
import numpy as np

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

set_seed(42)

if __name__ == "__main__":
    # 加载数据
    X, y = load_data()
    X = X.values if hasattr(X, 'values') else X

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    sc = StandardScaler()
    X_train_std = sc.fit_transform(X_train)
    X_test_std = sc.transform(X_test)

    y_test = pd.get_dummies(y_test).values.astype(np.float64)
    y_train = pd.get_dummies(y_train).values.astype(np.float64)


    layers = [
        Linear(input_size=64, output_size=128),
        ReLU(),
        Linear(input_size=128, output_size=256),
        ReLU(),
        # Linear(input_size=128, output_size=256),
        # ReLU(),
        Linear(input_size=256, output_size=10),
        Sigmoid()
    ]

    epochs = 100
    lr = 1e-1

    # mlp = MLPMultiClass(layers, epochs=epochs, lr=lr, input_shape=X_train.shape[1], output_shape=10)
    # mlp.train_MBGD(X_train, y_train)

    # loss_history = mlp.loss
    # length = len(loss_history)

    # # Plot the loss history
    # plt.figure(figsize=(10, 6))
    # plt.plot(range(length), loss_history, label='Training Loss')
    # plt.xlabel('Batches')
    # plt.ylabel('Loss')
    # plt.title('Training Loss over Batches')
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    # y_pred = mlp.predict(X_test)


    
