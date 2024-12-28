import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.preprocessing import load_data
from models.k_means import KMeansPlusPlus
from models.soft_k_means import SoftKMeans
from models.PCA import PCA
from models.autoencoder import AutoEncoder
from models.Neural import Linear, ReLU
from utils.random_seed import set_seed
from utils.align_labels import align_labels_hungarian

class DimensionalityReduction:
    def __init__(self, n_components: int = 2):
        self.n_components = n_components
        
    def perform_pca(self, X: np.ndarray) -> Tuple[np.ndarray, float]:
        """执行PCA降维并返回重构误差"""
        pca = PCA(n_components=self.n_components)
        X_pca = pca.fit_transform(X)
        X_recon_pca = pca.inverse_transform(X_pca)
        reconstruction_error = np.mean((X_recon_pca - X) ** 2)
        return X_pca, reconstruction_error
    
    def perform_autoencoder(self, X: np.ndarray, epochs: int = 5000, lr: float = 1e-2) -> Tuple[np.ndarray, float]:
        """执行自编码器降维并返回重构误差"""
        input_size = X.shape[1]
        nonlinear_layers = [
            Linear(input_size=input_size, output_size=self.n_components),
            ReLU(),
            Linear(self.n_components, self.n_components),
            Linear(input_size=self.n_components, output_size=input_size)
        ]
        
        ae = AutoEncoder(nonlinear_layers, epochs=epochs, lr=lr, 
                        input_shape=input_size, output_shape=input_size)
        ae.train_MBGD(X, X)
        
        X_ae = ae.encode(X)
        X_recon_ae = ae.predict(X)
        reconstruction_error = ae.get_loss(X_recon_ae, X)
        
        return X_ae, reconstruction_error

class ClusteringAnalysis:
    def __init__(self, n_clusters: int = 3):
        self.n_clusters = n_clusters
        
    def perform_clustering(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """执行K-means++和Soft K-means聚类"""
        kmeans = KMeansPlusPlus(n_clusters=self.n_clusters)
        soft_kmeans = SoftKMeans(n_clusters=self.n_clusters)
        
        kmeans.fit(X)
        soft_kmeans.fit(X)
        
        return kmeans.labels, soft_kmeans.labels

def visualize_results(X_pca: np.ndarray, X_ae: np.ndarray, 
                     kmeans_pca_labels: np.ndarray, soft_kmeans_pca_labels: np.ndarray,
                     kmeans_ae_labels: np.ndarray, soft_kmeans_ae_labels: np.ndarray,
                     pca_error: float, ae_error: float, true_labels: np.ndarray):
    """可视化聚类结果，并标记错误标签"""
    plt.figure(figsize=(15, 12))
    
    # PCA降维结果的聚类
    plt.subplot(221)
    error_mask = (kmeans_pca_labels != true_labels)
    plt.scatter(X_pca[~error_mask, 0], X_pca[~error_mask, 1], c=kmeans_pca_labels[~error_mask], cmap='viridis')
    plt.scatter(X_pca[error_mask, 0], X_pca[error_mask, 1], c='red', marker='x')
    plt.title(f'K-Means++ on PCA\nAccuracy: {(1-error_mask.mean())*100:.1f}%')
    
    plt.subplot(222)
    error_mask = (soft_kmeans_pca_labels != true_labels)
    plt.scatter(X_pca[~error_mask, 0], X_pca[~error_mask, 1], c=soft_kmeans_pca_labels[~error_mask], cmap='viridis')
    plt.scatter(X_pca[error_mask, 0], X_pca[error_mask, 1], c='red', marker='x')
    plt.title(f'Soft K-Means on PCA\nAccuracy: {(1-error_mask.mean())*100:.1f}%')
    
    # Autoencoder降维结果的聚类
    plt.subplot(223)
    error_mask = (kmeans_ae_labels != true_labels)
    plt.scatter(X_ae[~error_mask, 0], X_ae[~error_mask, 1], c=kmeans_ae_labels[~error_mask], cmap='viridis')
    plt.scatter(X_ae[error_mask, 0], X_ae[error_mask, 1], c='red', marker='x')
    plt.title(f'K-Means++ on Autoencoder\nAccuracy: {(1-error_mask.mean())*100:.1f}%')
    
    plt.subplot(224)
    error_mask = (soft_kmeans_ae_labels != true_labels)
    plt.scatter(X_ae[~error_mask, 0], X_ae[~error_mask, 1], c=soft_kmeans_ae_labels[~error_mask], cmap='viridis')
    plt.scatter(X_ae[error_mask, 0], X_ae[error_mask, 1], c='red', marker='x')
    plt.title(f'Soft K-Means on Autoencoder\nAccuracy: {(1-error_mask.mean())*100:.1f}%')
    
    plt.tight_layout()
    plt.show()

def main():
    set_seed(42)
    
    # 加载数据
    X, y = load_data()
    X = X.values if hasattr(X, 'values') else X
    
    # 初始化降维和聚类对象
    dim_reduction = DimensionalityReduction(n_components=2)
    clustering = ClusteringAnalysis(n_clusters=3)
    
    # 执行PCA降维和聚类
    X_pca, pca_error = dim_reduction.perform_pca(X)
    kmeans_pca_labels, soft_kmeans_pca_labels = clustering.perform_clustering(X_pca)
    print(f"PCA Reconstruction Error: {pca_error:.4f}")
    
    # 执行Autoencoder降维和聚类
    X_ae, ae_error = dim_reduction.perform_autoencoder(X)
    kmeans_ae_labels, soft_kmeans_ae_labels = clustering.perform_clustering(X_ae)
    print(f"Autoencoder Reconstruction Error: {ae_error:.4f}")


    # 对齐标签
    kmeans_pca_labels, _ = align_labels_hungarian(kmeans_pca_labels, y)
    soft_kmeans_pca_labels, _ = align_labels_hungarian(soft_kmeans_pca_labels, y)
    kmeans_ae_labels, _ = align_labels_hungarian(kmeans_ae_labels, y)
    soft_kmeans_ae_labels, _ = align_labels_hungarian(soft_kmeans_ae_labels, y)
    
    # 可视化结果
    visualize_results(X_pca, X_ae, 
                     kmeans_pca_labels, soft_kmeans_pca_labels,
                     kmeans_ae_labels, soft_kmeans_ae_labels,
                     pca_error, ae_error, y)
    

if __name__ == "__main__":
    main()