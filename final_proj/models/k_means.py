import numpy as np
from typing import Tuple, List
import numpy.typing as npt

class KMeansPlusPlus:
    def __init__(self, n_clusters: int = 3, max_iter: int = 100, random_state: int = None):
        """
        初始化 K-means++ 算法
        
        Args:
            n_clusters: 聚类数量
            max_iter: 最大迭代次数
            random_state: 随机种子
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.centroids = None
        self.labels_ = None
        
    def _init_centroids(self, X: npt.NDArray) -> npt.NDArray:
        """
        K-means++ 的初始化步骤
        
        Args:
            X: 输入数据，形状为 (n_samples, n_features)
            
        Returns:
            初始化的质心，形状为 (n_clusters, n_features)
        """
        n_samples, n_features = X.shape
        centroids = np.zeros((self.n_clusters, n_features))
        
        # 随机选择第一个质心
        centroids[0] = X[np.random.randint(n_samples)]

        for k in range(1, self.n_clusters):
            distances = []
            # 计算每个点到质心的距离
            for centroid in centroids[:k]:
                diff = X - centroid
                norms = np.linalg.norm(diff, axis=1)
                distances.append(norms)
            distances = np.array(distances).T
            min_distances = np.min(distances, axis=1) # 选取最小距离
            # 根据距离的平方作为概率选择新的质心
            prob = min_distances ** 2 / np.sum(min_distances ** 2)
            centroids[k] = X[np.random.choice(n_samples, p=prob)]
        
        return centroids
    
    def _compute_distances(self, X: npt.NDArray, centroids: npt.NDArray) -> npt.NDArray:
        """
        计算每个样本到所有质心的距离
        
        Args:
            X: 输入数据
            centroids: 当前质心
            
        Returns:
            距离矩阵，形状为 (n_samples, n_clusters)
        """
        distances = np.zeros((X.shape[0], self.n_clusters))
        for k in range(self.n_clusters): # 对距离矩阵的每一列进行计算，一列存储所有数据点对这个cluster中心的距离。
            distances[:, k] = np.linalg.norm(X - centroids[k], axis=1) # X - centroids[k]用了广播机制，每一行是一个差值向量，之后直接用norm对每一行进行计算
        return distances
    
    def _assign_clusters(self, distances: npt.NDArray) -> npt.NDArray:
        """
        根据距离分配聚类标签
        
        Args:
            distances: 距离矩阵
            
        Returns:
            聚类标签
        """
        return np.argmin(distances, axis=1)
    
    def _update_centroids(self, X: npt.NDArray, labels: npt.NDArray) -> npt.NDArray:
        """
        更新质心位置
        
        Args:
            X: 输入数据
            labels: 当前聚类标签
            
        Returns:
            更新后的质心
        """
        centroids = np.zeros((self.n_clusters, X.shape[1]))
        for k in range(self.n_clusters):
            mask = labels == k
            if np.any(mask):
                centroids[k] = X[mask].mean(axis=0)
        return centroids
    
    def fit(self, X: npt.NDArray) -> 'KMeansPlusPlus':
        """
        训练模型
        
        Args:
            X: 训练数据
            
        Returns:
            self
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)
            
        # 初始化质心
        self.centroids = self._init_centroids(X)
        
        for _ in range(self.max_iter):
            # 计算距离
            distances = self._compute_distances(X, self.centroids)
            
            # 分配聚类标签
            new_labels = self._assign_clusters(distances)
            
            # 如果标签没有变化，则停止迭代
            if self.labels_ is not None and np.all(new_labels == self.labels_):
                break
                
            self.labels_ = new_labels
            
            # 更新质心
            self.centroids = self._update_centroids(X, self.labels_)
            
        return self
    
    def predict(self, X: npt.NDArray) -> npt.NDArray:
        """
        预测新数据的聚类标签
        
        Args:
            X: 输入数据
            
        Returns:
            聚类标签
        """
        distances = self._compute_distances(X, self.centroids)
        return self._assign_clusters(distances)

if __name__ == "__main__":
    import sys
    import os

    # 将项目根目录添加到系统路径
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

    from utils.preprocessing import load_data
    
    # 加载数据
    X, y = load_data()

    X = X.values if hasattr(X, 'values') else X
    
    # 初始化模型
    kmeans = KMeansPlusPlus(n_clusters=3, random_state=42)
    
    # 训练模型
    kmeans.fit(X)
    
    # 打印结果
    print("聚类标签:", kmeans.labels_)
    print("质心:", kmeans.centroids)