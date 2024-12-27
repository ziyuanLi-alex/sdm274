import numpy as np
from typing import Tuple, List
import numpy.typing as npt

class SoftKMeans:
    def __init__(self, 
                 n_clusters: int = 3, 
                 max_iter: int = 100, 
                 beta: float = 1.0,  # 温度参数，控制软分配的程度
                 epsilon: float = 1e-5,  # 收敛阈值
                 random_state: int = None):
        """
        初始化 Soft K-means 算法
        
        Args:
            n_clusters: 聚类数量
            max_iter: 最大迭代次数
            beta: 温度参数，控制分配的软硬程度（越大越硬）
            epsilon: 收敛阈值
            random_state: 随机种子
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.beta = beta
        self.epsilon = epsilon
        self.random_state = random_state
        self.centroids = None
        self.membership = None

    @property
    def labels(self) -> np.ndarray:
        """返回从1开始的聚类标签"""
        if self.membership is None:
            return None
        return np.argmax(self.membership, axis=1) + 1

    def _compute_distances(self, X: npt.NDArray) -> npt.NDArray:
        """
        计算每个数据点到所有簇中心的距离
        
        Args:
            X: 输入数据，形状为 (n_samples, n_features)
            
        Returns:
            距离矩阵，形状为 (n_samples, n_clusters)
        """
        distances = np.zeros((X.shape[0], self.n_clusters)) # 距离矩阵
        for k in range(self.n_clusters):
            diff = X - self.centroids[k]
            distances[:, k] = np.sum(diff ** 2, axis=1)  # 距离矩阵的第K列是第K个簇中心到所有点的距离。这里使用了欧氏距离的平方
        return distances

    def _update_membership(self, X: npt.NDArray, update_internal: bool = True) -> npt.NDArray:
        """
        基于指数距离更新归属度矩阵
        
        Args:
            X: 输入数据
            update_internal: 是否更新内部状态
            
        Returns:
            更新后的归属度矩阵
        """
        distances = self._compute_distances(X)
        exp_distances = np.exp(-self.beta * distances)
        membership = exp_distances / np.sum(exp_distances, axis=1, keepdims=True)
        
        if update_internal:
            self.membership = membership
            
        return membership

    def _compute_centroids(self, X: npt.NDArray) -> npt.NDArray:
        """
        更新簇中心：m_k = Σ_n r_k^(n)x^(n) / Σ_n r_k^(n)
        
        Args:
            X: 输入数据，形状为 (n_samples, n_features)
            
        Returns:
            更新后的簇中心，形状为 (n_clusters, n_features)
        """
        # 初始化质心数组
        self.centroids = np.zeros((self.n_clusters, X.shape[1])) # 不需要保留原有结果，因为我们只需要点的归属度和坐标
        
        for k in range(self.n_clusters):
            # 分子：所有点的加权和。归属度和X做点乘
            # 举例：当一个点对k的归属度为0.3，会用0.3和这个点的坐标相乘。这里计算所有点的这个乘积
            numerator = np.sum(self.membership[:, k][:, np.newaxis] * X, axis=0) # 这里让membership的第k行和X做点乘，然后求和。
            
            # 分母：权重和
            denominator = np.sum(self.membership[:, k])
            self.centroids[k] = numerator / denominator
            
        return self.centroids

    def fit(self, X: npt.NDArray) -> 'SoftKMeans':
        """
        训练模型
        
        Args:
            X: 训练数据
            
        Returns:
            self
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)
            
        # 随机初始化质心（从数据点中选择）
        idx = np.random.choice(X.shape[0], self.n_clusters, replace=False)
        self.centroids = X[idx].copy()
        
        for _ in range(self.max_iter):
            old_membership = self.membership.copy() if self.membership is not None else None
            
            # E步：更新归属度(expectation)
            self.membership = self._update_membership(X)
            
            # M步：更新质心(maximization)
            self.centroids = self._compute_centroids(X)
            
            # 检查收敛性
            if old_membership is not None:
                if np.abs(self.membership - old_membership).max() < self.epsilon:
                    break
                    
        return self

    def predict(self, X: npt.NDArray) -> npt.NDArray:
        """
        预测新数据的聚类标签（硬分配）
        
        Args:
            X: 输入数据
            
        Returns:
            聚类标签（从1开始）
        """
        # 不更新内部状态
        membership = self._update_membership(X, update_internal=False)
        return np.argmax(membership, axis=1) + 1

    def predict_membership(self, X: npt.NDArray) -> npt.NDArray:
        """
        预测新数据对每个簇的归属度
        
        Args:
            X: 输入数据
            
        Returns:
            归属度矩阵
        """
        # 不更新内部状态
        return self._update_membership(X, update_internal=False)
    

if __name__ == "__main__":
    import sys
    import os

    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from utils.preprocessing import load_data
    from utils.align_labels import align_labels_hungarian
    
    # 加载数据
    X, y = load_data()
    X = X.values if hasattr(X, 'values') else X
    
    # 初始化模型
    soft_kmeans = SoftKMeans(n_clusters=3, random_state=42)
    
    # 训练模型
    soft_kmeans.fit(X)
    
    # 打印结果
    print("聚类标签:", soft_kmeans.predict(X))
    # print("归属度矩阵:", soft_kmeans.membership)
    print("质心:", soft_kmeans.centroids)

    # 调整标签以匹配真实标签
    aligned_labels, accuracy= align_labels_hungarian(y, soft_kmeans.predict(X))
    print("调整后的标签:", aligned_labels[:20])
    print("真实标签:", y[:20].values)
    print("准确率:", accuracy)


    