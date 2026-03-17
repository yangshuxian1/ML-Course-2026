# 实验二 模块2：KNN分类（不同K值对比）- 修复中文乱码
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# ========== 修复中文乱码核心配置 ==========
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号
# =========================================

# 1. 构造二维训练数据（两类样本）
X = np.array([[1,2],[2,3],[3,3],[6,5],[7,7],[8,6]])  # 特征
y = np.array([0,0,0,1,1,1])  # 标签（0/1两类）

# 2. 生成网格点（用于绘制分类边界）
xx, yy = np.meshgrid(np.linspace(0, 10, 200),
                     np.linspace(0, 10, 200))
grid = np.c_[xx.ravel(), yy.ravel()]  # 转换为模型输入格式

# 3. 遍历不同K值，训练并可视化分类结果
for k in [1,3,5]:
    # 初始化并训练KNN模型
    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(X, y)
    
    # 对网格点预测，绘制分类边界
    Z = knn_model.predict(grid)
    Z = Z.reshape(xx.shape)
    
    # 绘图
    plt.figure(figsize=(6, 4))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")  # 分类区域
    plt.scatter(X[:,0], X[:,1], c=y, s=80, cmap="coolwarm", edgecolor="black")  # 训练样本
    plt.xlabel("特征X1", fontsize=12)
    plt.ylabel("特征X2", fontsize=12)
    plt.title(f"KNN分类结果 (K = {k})", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.show()

# 打印K值影响说明
print("="*30)
print("KNN分类结果说明")
print("K=1：分类边界最复杂，易过拟合")
print("K=3：边界平滑度适中，平衡过拟合/欠拟合")
print("K=5：边界最平滑，泛化能力更强")
print("="*30)