import numpy as np
from typing import Optional

class DecisionStump:
    """决策树桩 - 专门为AdaBoost设计的深度为1的决策树"""
    
    class Node:
        def __init__(self) -> None:
            self.value = None  # 叶节点的预测值 (-1 或 1)
            self.feature_index = None  # 特征索引
            self.children = {}  # 子节点字典
            
    def __init__(self, gain_threshold: float = 1e-2) -> None:
        self.gain_threshold = gain_threshold
        
    def _weighted_entropy(self, y: np.ndarray, sample_weight: np.ndarray) -> float:
        """
        计算带权重的熵
        
        Args:
            y: 标签数组
            sample_weight: 样本权重数组
            
        Returns:
            weighted_entropy: 带权重的熵值
        """
        # 计算每个类别的加权计数
        classes = np.unique(y)
        weighted_counts = np.zeros(len(classes))
        for i, c in enumerate(classes):
            weighted_counts[i] = np.sum(sample_weight[y == c])
            
        # 计算概率
        total_weight = np.sum(sample_weight)
        probs = weighted_counts / total_weight
        
        # 计算熵
        nonzero_probs = probs[probs > 0]
        entropy = -np.sum(nonzero_probs * np.log2(nonzero_probs))
        return entropy
    
    def _weighted_conditional_entropy(self, feature: np.ndarray, y: np.ndarray, 
                                   sample_weight: np.ndarray) -> float:
        """
        计算带权重的条件熵
        
        Args:
            feature: 特征数组
            y: 标签数组
            sample_weight: 样本权重数组
            
        Returns:
            weighted_conditional_entropy: 带权重的条件熵值
        """
        feature_values = np.unique(feature)
        cond_entropy = 0.0
        total_weight = np.sum(sample_weight)
        
        for value in feature_values:
            mask = feature == value
            y_sub = y[mask]
            weights_sub = sample_weight[mask]
            
            if len(y_sub) > 0:
                value_prob = np.sum(weights_sub) / total_weight
                cond_entropy += value_prob * self._weighted_entropy(y_sub, weights_sub)
                
        return cond_entropy
    
    def _information_gain(self, feature: np.ndarray, y: np.ndarray, 
                         sample_weight: np.ndarray) -> float:
        """
        计算带权重的信息增益
        
        Args:
            feature: 特征数组
            y: 标签数组
            sample_weight: 样本权重数组
            
        Returns:
            information_gain: 信息增益值
        """
        return (self._weighted_entropy(y, sample_weight) - 
                self._weighted_conditional_entropy(feature, y, sample_weight))
    
    def _select_feature(self, X: np.ndarray, y: np.ndarray, 
                       sample_weight: np.ndarray) -> Optional[int]:
        """
        选择最佳分割特征
        
        Args:
            X: 特征矩阵
            y: 标签数组
            sample_weight: 样本权重数组
            
        Returns:
            best_feature_index: 最佳特征的索引
        """
        gains = [self._information_gain(X[:, i], y, sample_weight) 
                for i in range(X.shape[1])]
        best_idx = np.argmax(gains)
        
        if gains[best_idx] > self.gain_threshold:
            return best_idx
        return None
    
    def _weighted_majority_vote(self, y: np.ndarray, sample_weight: np.ndarray) -> int:
        """
        计算带权重的多数投票结果
        
        Args:
            y: 标签数组
            sample_weight: 样本权重数组
            
        Returns:
            majority_label: 权重最大的类别
        """
        unique_labels = np.unique(y)
        weighted_counts = np.array([np.sum(sample_weight[y == label]) 
                                  for label in unique_labels])
        return unique_labels[np.argmax(weighted_counts)]
    
    def fit(self, X: np.ndarray, y: np.ndarray, 
            sample_weight: Optional[np.ndarray] = None) -> 'DecisionStump':
        """
        训练决策树桩
        
        Args:
            X: 特征矩阵
            y: 标签数组
            sample_weight: 样本权重数组，如果为None则使用均匀权重
            
        Returns:
            self
        """
        if sample_weight is None:
            sample_weight = np.ones(len(y)) / len(y)
            
        # 创建根节点
        self.tree_ = DecisionStump.Node()
        
        # 默认预测为加权多数类
        self.tree_.value = self._weighted_majority_vote(y, sample_weight)
        
        # 选择最佳特征
        best_feature = self._select_feature(X, y, sample_weight)
        
        if best_feature is not None:
            self.tree_.feature_index = best_feature
            feature_values = np.unique(X[:, best_feature])
            
            # 对每个特征值创建一个子节点
            for value in feature_values:
                mask = X[:, best_feature] == value
                if np.any(mask):
                    child = DecisionStump.Node()
                    child.value = self._weighted_majority_vote(y[mask], 
                                                            sample_weight[mask])
                    self.tree_.children[value] = child
                    
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测新样本的标签
        
        Args:
            X: 特征矩阵
            
        Returns:
            predictions: 预测标签数组
        """
        def _predict_one(x):
            node = self.tree_
            if node.children:
                child = node.children.get(x[node.feature_index])
                if child:
                    return child.value
            return node.value
        
        return np.array([_predict_one(x) for x in X])

if __name__ == "__main__":
    # 简单的测试代码
    from sklearn.model_selection import train_test_split
    
    # 创建一个简单的二分类数据集
    X = np.array([[1, 2], [2, 1], [2, 2], [3, 1]])
    y = np.array([-1, -1, 1, 1])
    
    # 分割数据
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, 
                                                       random_state=42)
    
    # 创建并训练决策树桩
    stump = DecisionStump()
    # 使用带权重的训练
    sample_weights = np.array([0.4, 0.6])
    stump.fit(X_train, y_train, sample_weights)
    
    # 预测并打印结果
    predictions = stump.predict(X_test)
    print("预测结果:", predictions)