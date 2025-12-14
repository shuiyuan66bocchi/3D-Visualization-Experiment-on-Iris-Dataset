from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 加载数据并选择前两个类别
iris = load_iris()
X = iris.data[:, :3]
y = iris.target
X_binary = X[y != 2]
y_binary = y[y != 2]

# 训练模型
model = LogisticRegression(max_iter=200)
model.fit(X_binary, y_binary)

# 获取模型参数
coef = model.coef_[0]
intercept = model.intercept_[0]

# 创建3D图
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 绘制数据点
scatter = ax.scatter(X_binary[:, 0], X_binary[:, 1], X_binary[:, 2],
                     c=y_binary, cmap='coolwarm', s=50, alpha=0.8, edgecolor='k')

# 创建决策边界平面
x1_range = np.linspace(X_binary[:, 0].min()-0.5, X_binary[:, 0].max()+0.5, 20)
x2_range = np.linspace(X_binary[:, 1].min()-0.5, X_binary[:, 1].max()+0.5, 20)
x1_grid, x2_grid = np.meshgrid(x1_range, x2_range)

# 计算决策平面：w1*x1 + w2*x2 + w3*x3 + b = 0
x3_plane = -(coef[0] * x1_grid + coef[1] * x2_grid + intercept) / coef[2]

# 绘制决策平面
ax.plot_surface(x1_grid, x2_grid, x3_plane, alpha=0.5, color='gray')

# 设置标签
ax.set_xlabel('Sepal Length')
ax.set_ylabel('Sepal Width')
ax.set_zlabel('Petal Length')
ax.set_title(f'3D Decision Boundary\nDecision Plane: {coef[0]:.2f}x1 + {coef[1]:.2f}x2 + {coef[2]:.2f}x3 + {intercept:.2f} = 0')

plt.tight_layout()
plt.show()