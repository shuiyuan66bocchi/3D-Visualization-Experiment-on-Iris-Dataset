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

# 创建3D概率图
fig = plt.figure(figsize=(14, 6))

# 子图1：3D概率热图
ax1 = fig.add_subplot(121, projection='3d')

# 在x1-x2平面上创建网格，固定x3为中间值
x1_range = np.linspace(X_binary[:, 0].min()-0.5, X_binary[:, 0].max()+0.5, 30)
x2_range = np.linspace(X_binary[:, 1].min()-0.5, X_binary[:, 1].max()+0.5, 30)
x1_grid, x2_grid = np.meshgrid(x1_range, x2_range)
x3_fixed = np.mean(X_binary[:, 2])

# 计算该平面上的概率
X_plane = np.column_stack([x1_grid.ravel(), x2_grid.ravel(), 
                           np.full(x1_grid.ravel().shape[0], x3_fixed)])
prob_plane = model.predict_proba(X_plane)[:, 1].reshape(x1_grid.shape)

# 绘制概率曲面（高度表示概率值）
surf = ax1.plot_surface(x1_grid, x2_grid, prob_plane, cmap='RdYlBu_r', 
                       alpha=0.8, linewidth=0, antialiased=True)

# 绘制决策边界（p=0.5的等高线）
contour = ax1.contour(x1_grid, x2_grid, prob_plane, levels=[0.5], 
                     colors='black', linewidths=2, offset=x3_fixed)

# 绘制数据点
scatter1 = ax1.scatter(X_binary[:, 0], X_binary[:, 1], 
                       np.full(X_binary.shape[0], x3_fixed),
                       c=y_binary, cmap='coolwarm', s=30, alpha=0.6)

ax1.set_xlabel('Sepal Length')
ax1.set_ylabel('Sepal Width')
ax1.set_zlabel('Probability')
ax1.set_title(f'3D Probability Surface at Petal Length = {x3_fixed:.2f}')
ax1.view_init(elev=25, azim=45)

# 添加颜色条
fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=5, label='P(Class 1)')

# 子图2：概率等值面
ax2 = fig.add_subplot(122, projection='3d')

# 在多个x3值上绘制概率等高线
x3_values = np.linspace(X_binary[:, 2].min(), X_binary[:, 2].max(), 5)
colors = plt.cm.RdYlBu_r(np.linspace(0, 1, len(x3_values)))

for x3_val, color in zip(x3_values, colors):
    # 创建网格
    X_plane = np.column_stack([x1_grid.ravel(), x2_grid.ravel(), 
                              np.full(x1_grid.ravel().shape[0], x3_val)])
    prob_plane = model.predict_proba(X_plane)[:, 1].reshape(x1_grid.shape)
    
    # 绘制p=0.5的等高线（决策边界）
    ax2.contour(x1_grid, x2_grid, prob_plane, levels=[0.5], 
               colors=[color], linewidths=2, offset=x3_val)

# 绘制所有数据点
scatter2 = ax2.scatter(X_binary[:, 0], X_binary[:, 1], X_binary[:, 2],
                      c=y_binary, cmap='coolwarm', s=50, alpha=0.8, edgecolor='k')

ax2.set_xlabel('Sepal Length')
ax2.set_ylabel('Sepal Width')
ax2.set_zlabel('Petal Length')
ax2.set_title('Decision Boundaries at Different Petal Lengths')
ax2.view_init(elev=25, azim=45)

plt.tight_layout()
plt.show()

# 打印模型信息
print("="*50)
print("逻辑回归模型参数:")
print(f"系数: [{model.coef_[0][0]:.3f}, {model.coef_[0][1]:.3f}, {model.coef_[0][2]:.3f}]")
print(f"截距: {model.intercept_[0]:.3f}")
print(f"决策边界: {model.coef_[0][0]:.3f}x1 + {model.coef_[0][1]:.3f}x2 + {model.coef_[0][2]:.3f}x3 + {model.intercept_[0]:.3f} = 0")
print("="*50)