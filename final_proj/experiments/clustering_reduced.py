import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.preprocessing import load_data
from models.k_means import KMeansPlusPlus
from models.soft_k_means import SoftKMeans
from models.PCA import PCA
from models.autoencoder import AutoEncoder
from models.Neural import *
from utils.random_seed import set_seed

import numpy as np

import matplotlib.pyplot as plt

set_seed(42)

if __name__ == "__main__":
    # 加载数据
    X, y = load_data()
    X = X.values if hasattr(X, 'values') else X


    # 降维部分
    # 1. PCA降维
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    X_recon_pca = pca.inverse_transform(X_pca)
    print("PCA Reconstruction Error:", np.mean((X_recon_pca - X) ** 2))

    # 2. Autoencoder降维
    nonlinear_layers = [
        Linear(input_size=7, output_size=2),
        ReLU(),
        Linear(2,2),
        Linear(input_size=2, output_size=7)
    ]
    ae = AutoEncoder(nonlinear_layers, epochs=5000, lr=1e-2, input_shape=7, output_shape=7)
    ae.train_BGD(X, X)
    X_ae = ae.encode(X)
    X_recon_ae = ae.predict(X)
    print("Autoencoder Reconstruction Error:", ae.get_loss(X_recon_ae, X))

    # 聚类部分
    # 1. PCA降维后的聚类
    kmeans_pca = KMeansPlusPlus(n_clusters=3)
    soft_kmeans_pca = SoftKMeans(n_clusters=3)
    kmeans_pca.fit(X_pca)
    soft_kmeans_pca.fit(X_pca)

    # 2. Autoencoder降维后的聚类
    kmeans_ae = KMeansPlusPlus(n_clusters=3)
    soft_kmeans_ae = SoftKMeans(n_clusters=3)
    kmeans_ae.fit(X_ae)
    soft_kmeans_ae.fit(X_ae)


    # 初始化聚类模型
    kmeans_pca = KMeansPlusPlus(n_clusters=3)
    kmeans_ae = KMeansPlusPlus(n_clusters=3)
    soft_kmeans_pca = SoftKMeans(n_clusters=3)
    soft_kmeans_ae = SoftKMeans(n_clusters=3)
   

     # 2. 聚类结果可视化
    plt.figure(figsize=(15, 10))
    
    # PCA降维结果的聚类
    plt.subplot(221)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_pca.labels)
    plt.title('K-Means++ on PCA')
    plt.colorbar()

    plt.subplot(222)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=soft_kmeans_pca.labels)
    plt.title('Soft K-Means on PCA')
    plt.colorbar()

    # Autoencoder降维结果的聚类
    plt.subplot(223)
    plt.scatter(X_ae[:, 0], X_ae[:, 1], c=kmeans_ae.labels)
    plt.title('K-Means++ on Autoencoder')
    plt.colorbar()

    plt.subplot(224)
    plt.scatter(X_ae[:, 0], X_ae[:, 1], c=soft_kmeans_ae.labels)
    plt.title('Soft K-Means on Autoencoder')
    plt.colorbar()
    
    
    plt.show()






    








