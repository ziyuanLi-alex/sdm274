import numpy as np
from typing import List, Tuple, Dict, Optional
import numpy.typing as npt
import sys
import os

# 将项目根目录添加到系统路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.Solver import Solver
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from utils.evaluation import Evaluation
from sklearn.metrics import classification_report



class OVOSVM:
    def __init__(self, 
                 C: float = 1.0,
                 kernel: str = 'linear',
                 gamma: float = 1.0,
                 max_iter: int = 1000,
                 tol: float = 1e-5,
                 random_state: int = None):
        """
        One-vs-One SVM分类器框架
        
        Args:
            C: 软间隔参数
            kernel: 核函数类型 ('linear' or 'rbf')
            gamma: RBF核参数
            max_iter: 最大迭代次数
            tol: 收敛阈值
            random_state: 随机种子
        """
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.binary_classifiers = {}
        
    def _compute_kernel(self, X1: npt.NDArray, X2: npt.NDArray) -> npt.NDArray:
        """计算核矩阵"""
        if self.kernel == 'linear':
            return np.dot(X1, X2.T)
        elif self.kernel == 'rbf':
            # 计算RBF核 K(x,y) = exp(-gamma||x-y||^2)
            squared_norm = np.sum(X1**2, axis=1).reshape(-1, 1) + \
                         np.sum(X2**2, axis=1) - \
                         2 * np.dot(X1, X2.T)
            return np.exp(-self.gamma * squared_norm)
        else:
            raise ValueError(f"Unsupported kernel type: {self.kernel}")
    
    def _train_binary_svm(self, X: npt.NDArray, y: npt.NDArray) -> Tuple:
        """
        训练二分类SVM（使用SMO算法）
        
        Args:
            X: 训练数据
            y: 训练标签 (+1/-1)
            
        Returns:
            (alpha, b, support_vectors, sv_y)
        """
        n_samples = X.shape[0]
        
        # 计算核矩阵
        K = self._compute_kernel(X, X)
        
        # 初始化SMO求解器
        p = -np.ones(n_samples)
        solver = Solver(Q=K, 
                       p=p, 
                       y=y, 
                       C=self.C, 
                       tol=self.tol)
        
        # SMO迭代优化
        for _ in range(self.max_iter):
            i, j = solver.working_set_select()
            if i < 0:  # 收敛条件
                break
            solver.update(i, j)
            
        # 获取支持向量
        support_mask = solver.alpha > self.tol
        return (solver.alpha[support_mask],
                solver.calculate_rho(),
                X[support_mask],
                y[support_mask])
    
    def fit(self, X: npt.NDArray, y: npt.NDArray) -> 'OVOSVM':
        """
        训练OVO-SVM分类器
        
        Args:
            X: 训练数据
            y: 训练标签
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)
            
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        
        # 为每一对类别训练一个二分类器
        for i in range(n_classes):
            for j in range(i + 1, n_classes):
                # 选择当前两个类别的数据
                mask = (y == self.classes[i]) | (y == self.classes[j])
                X_sub = X[mask]
                y_sub = y[mask]
                
                # 转换标签为+1/-1
                y_sub = np.where(y_sub == self.classes[i], 1, -1)
                
                # 训练二分类器
                classifier_params = self._train_binary_svm(X_sub, y_sub)
                self.binary_classifiers[(i, j)] = classifier_params
                
        return self
    
    def _predict_binary(self, x: npt.NDArray, classifier_params: Tuple) -> int:
        """
        使用二分类器进行预测
        
        Args:
            x: 输入样本
            classifier_params: (alpha, b, support_vectors, sv_y)
            
        Returns:
            预测类别 (+1/-1)
        """
        alpha, b, support_vectors, sv_y = classifier_params
        
        # 计算决策函数
        K = self._compute_kernel(support_vectors, x.reshape(1, -1))
        decision = np.sum(alpha * sv_y * K.reshape(-1)) - b
        return 1 if decision >= 0 else -1
    
    def predict(self, X: npt.NDArray) -> npt.NDArray:
        """
        对新数据进行预测
        
        Args:
            X: 测试数据
        """
        n_samples = X.shape[0]
        predictions = np.zeros((n_samples, len(self.classes)))
        
        # 对每个样本使用所有二分类器进行预测
        for idx, x in enumerate(X):
            for (i, j), classifier_params in self.binary_classifiers.items():
                pred = self._predict_binary(x, classifier_params)
                if pred == 1:
                    predictions[idx, i] += 1
                else:
                    predictions[idx, j] += 1
        
        # 返回得票最多的类别
        return self.classes[np.argmax(predictions, axis=1)]

if __name__ == "__main__":
    from utils.preprocessing import load_data
    
    # 加载数据
    X, y = load_data()
    X = X.values if hasattr(X, 'values') else X

    # 数据预处理
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    # 创建不同核函数的SVM分类器
    classifiers = {
        'Linear SVM': OVOSVM(C=1.0, kernel='linear'),
        'RBF SVM': OVOSVM(C=1.0, kernel='rbf', gamma=0.1)
    }

    eval = Evaluation()

    for name, clf in classifiers.items():
        print(f"\nTraining {name}...")
        
        # 训练模型
        clf.fit(X_train, y_train)
        
        # 在训练集上评估
        train_pred = clf.predict(X_train)
        # train_acc = accuracy_score(y_train, train_pred)
        train_acc = eval.calculate_accuracy(y_train, train_pred)
        print(f"{name} Training Accuracy: {train_acc:.4f}")
        
        # 在测试集上评估
        test_pred = clf.predict(X_test)
        # test_acc = accuracy_score(y_test, test_pred)
        test_acc = eval.calculate_accuracy(y_test, test_pred)
        print(f"{name} Test Accuracy: {test_acc:.4f}")
        
        # 输出详细的分类报告
        print(f"\n{name} Classification Report:")
        print(classification_report(y_test, test_pred))

    demo_model = classifiers['RBF SVM']  # 假设RBF核表现最好
    
    # 选择一个测试样本
    sample = X_test[0:1]  # 取第一个测试样本
    prediction = demo_model.predict(sample)
    true_label = y_test.iloc[0]
    
    print("\nSingle Sample Prediction Example:")
    print(f"True label: {true_label}")
    print(f"Predicted label: {prediction[0]}")