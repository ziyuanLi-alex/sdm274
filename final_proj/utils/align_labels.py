import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import Tuple
import numpy.typing as npt

def align_labels_hungarian(pred_labels: npt.NDArray, true_labels: npt.NDArray) -> Tuple[npt.NDArray, float]:
    """
    使用匈牙利算法将预测的聚类标签与真实标签对齐
    
    Args:
        pred_labels: 预测的聚类标签（从0开始或1开始）
        true_labels: 真实标签（从0开始或1开始）
        
    Returns:
        aligned_labels: 重新映射后的预测标签
        accuracy: 对齐后的准确率
    """
    # 确保标签从0开始
    offset = true_labels.min()
    true_labels = true_labels - offset
    pred_labels = pred_labels - pred_labels.min()
    
    # 获取类别数
    n_classes = max(len(np.unique(true_labels)), len(np.unique(pred_labels)))
    
    # 构建成本矩阵（负的共现矩阵）
    cost_matrix = np.zeros((n_classes, n_classes))
    
    # 填充成本矩阵
    for i in range(n_classes):
        for j in range(n_classes):
            # 计算标签i和j的共现次数的负值
            cost_matrix[i, j] = -np.sum((pred_labels == i) & (true_labels == j))
    
    # 使用匈牙利算法找到最优匹配
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # 创建标签映射
    label_mapping = dict(zip(row_ind, col_ind))
    
    # 应用映射到预测标签
    aligned_labels = np.array([label_mapping[label] for label in pred_labels])
    
    # 恢复原始标签范围
    aligned_labels = aligned_labels + offset
    
    # 计算对齐后的准确率
    accuracy = np.mean(aligned_labels == true_labels + offset)
    
    return aligned_labels, accuracy