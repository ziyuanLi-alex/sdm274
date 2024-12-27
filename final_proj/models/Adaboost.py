import numpy as np
from typing import List
import numpy.typing as npt
import sys
import os

# 将项目根目录添加到系统路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from sklearn.preprocessing import StandardScaler
from utils.evaluation import Evaluation
from sklearn.metrics import classification_report
from models.DecisonStump import DecisionStump
from utils.random_seed import set_seed

set_seed(42)


class AdaBoost:
    def __init__(self, n_estimators: int = 50, learning_rate: float = 1.0):
        """
        初始化AdaBoost分类器
        
        Args:
            n_estimators: 基学习器的数量
            learning_rate: 学习率
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.estimators: List[DecisionStump] = []
        self.estimator_weights = np.zeros(n_estimators)
        
    def fit(self, X: npt.NDArray, y: npt.NDArray) -> 'AdaBoost':
        """
        训练AdaBoost分类器
        
        Args:
            X: 训练数据
            y: 训练标签 (-1 或 1)
            
        Returns:
            self
        """
        n_samples = X.shape[0]
        # 初始化样本权重
        sample_weight = np.ones(n_samples) / n_samples
        
        for i in range(self.n_estimators):
            # 训练决策树桩
            estimator = DecisionStump()
            estimator.fit(X, y, sample_weight)
            
            # 获取预测结果
            predictions = estimator.predict(X)
            
            # 计算加权错误率
            incorrect = predictions != y
            error = np.sum(sample_weight * incorrect) / np.sum(sample_weight)
            
            # 如果错误率为0或1，停止训练
            if error == 0 or error >= 1.0:
                break
                
            # 计算这个分类器的权重
            estimator_weight = self.learning_rate * np.log((1 - error) / error)
            
            # 保存分类器和权重
            self.estimators.append(estimator)
            self.estimator_weights[i] = estimator_weight
            
            # 更新样本权重
            sample_weight *= np.exp(estimator_weight * incorrect)
            sample_weight /= np.sum(sample_weight)  # 归一化
            
        return self
    
    def predict(self, X: npt.NDArray) -> npt.NDArray:
        """
        预测新样本的标签
        
        Args:
            X: 测试数据
            
        Returns:
            predictions: 预测标签 (-1 或 1)
        """
        n_samples = X.shape[0]
        predictions = np.zeros(n_samples)
        
        # 累积所有分类器的加权预测
        for estimator, weight in zip(self.estimators, self.estimator_weights):
            predictions += weight * estimator.predict(X)
            
        # 返回符号
        return np.sign(predictions)

if __name__ == "__main__":
    from utils.preprocessing import load_data
    from sklearn.model_selection import train_test_split
    
    # 加载数据
    X, y = load_data()
    X = X.values if hasattr(X, 'values') else X
    
    # 创建二分类问题（移除标签为2的样本）
    mask = y != 2
    X = X[mask]
    y = y[mask]
    # 将标签转换为{-1, 1}
    y = np.where(y == 1, 1, -1)
    
    # 分割数据
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 训练模型
    model = AdaBoost(n_estimators=50)
    model.fit(X_train, y_train)
    
    eval = Evaluation()

    # 预测并评估
    y_pred = model.predict(X_test)
    # accuracy = accuracy_score(y_test, y_pred)
    accuracy = eval.calculate_accuracy(y_test, y_pred)
    print(f"测试集准确率: {accuracy:.4f}")