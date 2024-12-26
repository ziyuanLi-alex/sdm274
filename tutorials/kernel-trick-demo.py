import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import make_circles
from sklearn.preprocessing import StandardScaler

# Create a dataset that's not linearly separable (concentric circles)
X, y = make_circles(n_samples=100, noise=0.15, factor=0.5)
X = StandardScaler().fit_transform(X)

# Create figure
plt.figure(figsize=(12, 5))

# Plot 1: Linear SVM (without kernel trick)
plt.subplot(121)
svm_linear = SVC(kernel='linear')
svm_linear.fit(X, y)

# Create mesh grid for decision boundary
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

# Plot decision boundary
Z = svm_linear.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.4)
plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
plt.title('Linear SVM\nWithout Kernel Trick')

# Plot 2: RBF Kernel SVM (with kernel trick)
plt.subplot(122)
svm_rbf = SVC(kernel='rbf')
svm_rbf.fit(X, y)

# Plot decision boundary
Z = svm_rbf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.4)
plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
plt.title('RBF Kernel SVM\nWith Kernel Trick')

plt.tight_layout()
plt.show()

# Print number of support vectors used in each case
print(f"Number of support vectors (Linear SVM): {len(svm_linear.support_vectors_)}")
print(f"Number of support vectors (RBF Kernel SVM): {len(svm_rbf.support_vectors_)}")
