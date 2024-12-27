import numpy as np
import sys
import matplotlib.pyplot as plt
import os

# 将项目根目录添加到系统路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.Neural import MLP, Activation, Sigmoid, ReLU, Linear

class AutoEncoder(MLP):
    def __init__(self, layers, epochs=500, lr=0.01, input_shape=13, output_shape=13):
        
        super().__init__(layers, epochs, lr, input_shape, output_shape)
        self.threshold = 0.5
        
    def get_loss(self, x_pred, x_true):
        x_pred = x_pred.reshape(-1, self.output_shape)
        x_true = x_true.reshape(-1, self.output_shape)

        # MSE loss for reconstruction
        return np.mean((x_pred - x_true) ** 2)


    def get_loss_grad(self, x_pred, x_true):
        # The loss here is similar to MSE, and the grad is alike.
        x_pred = x_pred.reshape(-1, self.output_shape)
        x_true = x_true.reshape(-1, self.output_shape)

        return (x_pred - x_true) / self.output_shape

    
    def predict(self, X):
        # Forward pass
        x_recon = self.forward(X)
        
        # Convert the output to binary
        return x_recon
    
    def _find_encoder_end(self):
        """找到编码器的终止位置（即输出维度最小的位置）"""
        min_dim = float('inf')
        encoder_end = 0
        current_dim = self.input_shape
        
        for i, layer in enumerate(self.layers):
            if isinstance(layer, Linear):
                current_dim = np.shape(layer.W)[1]
                if current_dim < min_dim:
                    min_dim = current_dim
                    encoder_end = i + 1
        return encoder_end
    
    def encode(self, X):
        """只运行编码器部分，获取降维结果"""
        out = X
        # 只运行到encoder_end的位置
        for layer in self.layers[:self._find_encoder_end()]:
            out = layer.forward(out)
        return out
    


def find_optimal_lr(X, lr_range=[1e-5, 1e-4, 1e-3, 1e-2, 1e-1]):
    best_lr = None
    best_loss = float('inf')
    losses = []
    
    for lr in lr_range:
        # 使用相同的网络结构
        nonlinear_layers = [
            Linear(input_size=7, output_size=2),
            ReLU(),
            Linear(2,2),
            Linear(input_size=2, output_size=7)
        ]
        
        # 用较少的epoch来快速测试
        model = AutoEncoder(nonlinear_layers, epochs=5000, lr=lr, 
                          input_shape=7, output_shape=7)
        model.train_BGD(X, X)
        
        # 记录重构误差
        loss = model.get_loss(model.predict(X), X)
        losses.append(loss)
        
        if loss < best_loss:
            best_loss = loss
            best_lr = lr
            
        print(f"Learning rate: {lr}, Loss: {loss}")
    
    return best_lr, losses

def find_optimal_epochs(X, lr, epoch_range=[1000, 2000, 5000, 10000, 20000]):
    losses_history = []
    best_epochs = None
    best_loss = float('inf')
    
    for epochs in epoch_range:
        nonlinear_layers = [
            Linear(input_size=7, output_size=2),
            ReLU(),
            Linear(2,2),
            Linear(input_size=2, output_size=7)
        ]
        
        model = AutoEncoder(nonlinear_layers, epochs=epochs, lr=lr, 
                          input_shape=7, output_shape=7)
        
        # 记录训练过程中的损失
        loss_history = []
        def callback(epoch, loss):
            if epoch % 100 == 0:
                loss_history.append(loss)
        
        model.train_BGD(X, X)
        
        final_loss = model.get_loss(model.predict(X), X)
        losses_history.append(loss_history)
        
        if final_loss < best_loss:
            best_loss = final_loss
            best_epochs = epochs
            
        print(f"Epochs: {epochs}, Final Loss: {final_loss}")
    
    return best_epochs, losses_history

if __name__ == "__main__":
    import sys
    import os

    # 将项目根目录添加到系统路径
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

    from utils.preprocessing import load_data
    
    # 加载数据
    X, y = load_data()

    X = X.values if hasattr(X, 'values') else X

    # best_lr, losses = find_optimal_lr(X, lr_range=[1e-5, 1e-4, 1e-3, 1e-2, 1e-1])
    # best_epochs, losses_history = find_optimal_epochs(X, 1e-2, epoch_range=[1000, 2000, 5000, 10000, 20000])

    nonlinear_layers = [
        Linear(input_size=7, output_size=2),
        ReLU(),
        Linear(2,2),
        # ReLU(),
        Linear(input_size=2, output_size=7)
    ]

    epochs = 5000
    lr = 1e-2

    nonlin_autoencoder = AutoEncoder(nonlinear_layers, epochs=epochs, lr=lr, input_shape=7, output_shape=7)
    nonlin_autoencoder.train_BGD(X, X)

    # Implement another autoencoder with a middle layer of 3 dimensions
    nonlinear_layers_3d = [
        Linear(input_size=7, output_size=3),
        ReLU(),
        Linear(3, 3),
        Linear(input_size=3, output_size=7)
    ]

    autoencoder_3d = AutoEncoder(nonlinear_layers_3d, epochs=10000, lr=lr, input_shape=7, output_shape=7)
    autoencoder_3d.train_BGD(X, X)

    # X_recon_3d = autoencoder_3d.predict(X)
    # print(X_recon_3d[:5])
    # print(X[:5])
    # print("Reconstruction Error with 3D middle layer:", autoencoder_3d.get_loss(X_recon_3d, X))

    # X_recon = nonlin_autoencoder.predict(X)
    # print(X_recon[:1])
    # print(X[:1])
    # print("Reconstruction Error:" , nonlin_autoencoder.get_loss(X_recon, X))

    # Visualize the original data and reconstructed data using bar charts
    # fig, ax = plt.subplots(figsize=(12,9), nrows=4, ncols=3)

    # for i in range(1,5):
    #     for j in range(1,4):
    #         ax[i-1, j-1].bar(range(X.shape[1]), X[i*j-1], color='blue', alpha=0.5, label='Original Data')
    #         ax[i-1, j-1].bar(range(X_recon.shape[1]), X_recon[i*j-1], color='red', alpha=0.5, label='Reconstructed Data')
    #         ax[i-1, j-1].set_xlabel('Features')
    #         ax[i-1, j-1].set_ylabel('Values')

    # plt.show()

    # print(X[:5])
    # print(X_recon[:5])

    # Visualize the encoded data points
    encoded_data = nonlin_autoencoder.encode(X)
    encoded_data_3d = autoencoder_3d.encode(X)

    import matplotlib.pyplot as plt

    # 可视化转换后的数据点
    plt.figure(figsize=(8, 6))
    plt.scatter(encoded_data[:, 0], encoded_data[:, 1], c=y, cmap='viridis', edgecolor='k', s=50)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('Encoded Dataset (2 components)')
    plt.colorbar()
    plt.show()
    # 可视化3维转换后的数据点

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(encoded_data_3d[:, 0], encoded_data_3d[:, 1], encoded_data_3d[:, 2], c=y, cmap='viridis', edgecolor='k', s=50)
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_zlabel('Principal Component 3')
    ax.set_title('Encoded Dataset (3 components)')
    fig.colorbar(scatter)
    plt.show()









