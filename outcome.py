from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 加载数据
iris = load_iris()
X = iris.data[:, :3]
y = iris.target
X_binary = X[y != 2]
y_binary = y[y != 2]

# 训练模型
model = LogisticRegression(max_iter=200)
model.fit(X_binary, y_binary)
coef = model.coef_[0]
intercept = model.intercept_[0]

# 创建图形
fig = plt.figure(figsize=(14, 6))

# 任务二：3D Decision Boundary
ax1 = fig.add_subplot(121, projection='3d')

# 绘制数据点
ax1.scatter(X_binary[:, 0], X_binary[:, 1], X_binary[:, 2],
            c=y_binary, cmap='coolwarm', s=50, alpha=0.8, edgecolor='k')

# 绘制决策平面
x1_range = np.linspace(X_binary[:, 0].min()-0.5, X_binary[:, 0].max()+0.5, 20)
x2_range = np.linspace(X_binary[:, 1].min()-0.5, X_binary[:, 1].max()+0.5, 20)
x1_grid, x2_grid = np.meshgrid(x1_range, x2_range)
x3_plane = -(coef[0] * x1_grid + coef[1] * x2_grid + intercept) / coef[2]

ax1.plot_surface(x1_grid, x2_grid, x3_plane, alpha=0.5, color='gray')
ax1.set_xlabel('Sepal Length')
ax1.set_ylabel('Sepal Width')
ax1.set_zlabel('Petal Length')
ax1.set_title('Task 2: 3D Decision Boundary')

# 任务三：3D Probability Map
ax2 = fig.add_subplot(122, projection='3d')

# 创建概率网格
x1_range = np.linspace(X_binary[:, 0].min()-0.5, X_binary[:, 0].max()+0.5, 30)
x2_range = np.linspace(X_binary[:, 1].min()-0.5, X_binary[:, 1].max()+0.5, 30)
x1_grid, x2_grid = np.meshgrid(x1_range, x2_range)
x3_mid = np.mean(X_binary[:, 2])

# 计算概率
X_plane = np.column_stack([x1_grid.ravel(), x2_grid.ravel(), 
                          np.full(x1_grid.ravel().shape[0], x3_mid)])
prob_plane = model.predict_proba(X_plane)[:, 1].reshape(x1_grid.shape)

# 绘制概率曲面
surf = ax2.plot_surface(x1_grid, x2_grid, prob_plane, cmap='RdYlBu_r', 
                       alpha=0.8, linewidth=0)

# 绘制数据点
ax2.scatter(X_binary[:, 0], X_binary[:, 1], 
           np.full(X_binary.shape[0], x3_mid),
           c=y_binary, cmap='coolwarm', s=30, alpha=0.6)

ax2.set_xlabel('Sepal Length')
ax2.set_ylabel('Sepal Width')
ax2.set_zlabel('Probability')
ax2.set_title('Task 3: 3D Probability Map')

# 添加概率颜色条
fig.colorbar(surf, ax=ax2, shrink=0.5, aspect=5, label='P(Class 1)')

plt.tight_layout()
plt.show()