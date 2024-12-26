import numpy as np
import numpy.typing as npt
from typing import Tuple, Optional


# Components: PCA转换后的主成分
# Features: 原来数据的特征

class PCA:
    def __init__(self, n_components: int = None, random_state: Optional[int] = None):
        """
        初始化PCA类
        
        Args:
            n_components: 需要保留的主成分数量。如果为None，则保留所有成分
            random_state: 随机种子，用于结果复现
        """
        self.n_components = n_components
        self.random_state = random_state
        
        # 将在fit过程中计算的属性
        self.components_ = None  # 主成分向量
        self.explained_variance_ = None  # 每个主成分的方差
        self.explained_variance_ratio_ = None  # 每个主成分解释的方差比例
        self.mean_ = None  # 训练数据的均值
        self.n_features_ = None  # 特征数量
        self.n_samples_ = None  # 样本数量
        
    def fit(self, X: npt.NDArray) -> 'PCA':
        """
        训练PCA模型
        
        Args:
            X: 形状为 (n_samples, n_features) 的训练数据
            
        Returns:
            self: 训练后的PCA实例
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)
            
        self.n_samples_, self.n_features_ = X.shape
        
        # 计算并减去均值
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        
        # 计算协方差矩阵
        cov_matrix = np.cov(X_centered.T)
        
        # 计算特征值和特征向量
        eigenvals, eigenvecs = np.linalg.eig(cov_matrix)
        
        # 将特征值和对应的特征向量按特征值大小排序
        idx = eigenvals.argsort()[::-1]
        eigenvals = eigenvals[idx]
        eigenvecs = eigenvecs[:, idx]
        
        # 确定要保留的主成分数量
        if self.n_components is None:
            self.n_components = self.n_features_
            
        # 保存主成分和相关统计量
        self.components_ = eigenvecs[:, :self.n_components] # 取前n_components列。这里会发生信息损失
        self.explained_variance_ = eigenvals[:self.n_components]
        total_var = eigenvals.sum()
        self.explained_variance_ratio_ = eigenvals[:self.n_components] / total_var
        
        return self
    
    def transform(self, X: npt.NDArray) -> npt.NDArray:
        """
        使用训练好的PCA模型将数据转换到新的特征空间
        
        Args:
            X: 形状为 (n_samples, n_features) 的输入数据
            
        Returns:
            形状为 (n_samples, n_components) 的转换后的数据
        """
        # 检查特征数量是否匹配
        if X.shape[1] != self.n_features_:
            raise ValueError(f"期望的特征数量为 {self.n_features_}，但输入数据的特征数量为 {X.shape[1]}")
            
        # 中心化数据
        X_centered = X - self.mean_
        
        # 投影到主成分空间
        X_transformed = np.dot(X_centered, self.components_)
        
        return X_transformed
    
    def fit_transform(self, X: npt.NDArray) -> npt.NDArray:
        """
        训练PCA模型并转换数据
        
        Args:
            X: 形状为 (n_samples, n_features) 的输入数据
            
        Returns:
            形状为 (n_samples, n_components) 的转换后的数据
        """
        return self.fit(X).transform(X)
    
    def inverse_transform(self, X: npt.NDArray) -> npt.NDArray:
        """
        将降维后的数据转换回原始特征空间
        
        Args:
            X: 形状为 (n_samples, n_components) 的降维后的数据
            
        Returns:
            形状为 (n_samples, n_features) 的重构后的数据
        """
        if X.shape[1] != self.n_components:
            raise ValueError(f"期望的特征数量为 {self.n_components}，但输入数据的特征数量为 {X.shape[1]}")
            
        # 使用主成分重构原始空间的数据
        X_reconstructed = np.dot(X, self.components_.T) + self.mean_
        
        return X_reconstructed
    
    def get_feature_importance(self) -> Tuple[npt.NDArray, npt.NDArray]:
        """
        获取特征重要性
        
        Returns:
            feature_importance: 每个原始特征的重要性得分
            feature_components: 每个特征在各个主成分上的投影
        """
        # 计算特征重要性
        feature_importance = np.sum(np.abs(self.components_), axis=1)
        feature_importance /= np.sum(feature_importance)
        
        # 获取特征在主成分上的投影
        feature_components = self.components_.T
        
        return feature_importance, feature_components

if __name__ == "__main__":
    import sys
    import os

    # 将项目根目录添加到系统路径
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

    from utils.preprocessing import load_data
    
    # 加载数据
    X, y = load_data()

    X = X.values if hasattr(X, 'values') else X

    # 初始化PCA模型
    pca = PCA(n_components=2, random_state=42)
    X_transformed = pca.fit(X).transform(X)
    X_reconstructed = pca.inverse_transform(X_transformed)

    reconstruction_error = np.mean((X_reconstructed - X) ** 2)

    print('Reconstruction error:', reconstruction_error)

