# 实验二 模块3：梯度下降法（优化f(x)=x²+2x+1）- 修复中文乱码
import torch
import numpy as np
import matplotlib.pyplot as plt

# ========== 修复中文乱码核心配置 ==========
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号
# =========================================

# 1. 基础梯度下降：迭代优化参数
print("="*30)
print("梯度下降优化结果")
x = torch.tensor([5.0], requires_grad=True)  # 初始值x=5，开启梯度追踪
lr = 0.1  # 学习率（调优后，确保收敛）
for i in range(20):
    y = x**2 + 2*x + 1  # 目标函数（最小值在x=-1）
    y.backward()  # 反向传播计算梯度
    with torch.no_grad():  # 关闭梯度追踪，更新参数
        x -= lr * x.grad
    x.grad.zero_()  # 清空梯度，避免累加

# 打印最终结果
print(f"目标函数f(x)=x²+2x+1的最优解x = {x.item():.4f}")
print(f"最优解对应的损失值y = {y.item():.4f}")
print("="*30)

# 2. 可视化：损失曲线（迭代过程）
loss_list = []
x = torch.tensor([5.0], requires_grad=True)
for i in range(20):
    y = x**2 + 2*x + 1
    loss_list.append(y.item())  # 记录每次损失
    y.backward()
    with torch.no_grad():
        x -= 0.1 * x.grad
    x.grad.zero_()

# 绘制损失曲线
plt.figure(figsize=(8, 4))
plt.plot(loss_list, marker='o', markersize=4, color='blue', linewidth=2)
plt.xlabel("迭代次数", fontsize=12)
plt.ylabel("损失值y", fontsize=12)
plt.title("梯度下降 - 损失曲线 (学习率=0.1)", fontsize=14)
plt.grid(True, alpha=0.3)
plt.show()

# 3. 可视化：优化路径（参数逼近最优值的过程）
x_vals = np.linspace(-5, 5, 100)
y_vals = x_vals**2 + 2*x_vals + 1  # 目标函数曲线

x_path = []
x = torch.tensor([5.0], requires_grad=True)
lr = 0.3  # 增大学习率，路径更明显
for i in range(10):
    y = x**2 + 2*x + 1
    x_path.append(x.item())  # 记录每次x的取值
    y.backward()
    with torch.no_grad():
        x -= lr * x.grad
    x.grad.zero_()

# 绘制优化路径
plt.figure(figsize=(8, 6))
plt.plot(x_vals, y_vals, color='black', label="目标函数f(x)=x²+2x+1", linewidth=2)
plt.scatter(x_path, [xx**2 + 2*xx + 1 for xx in x_path], 
            color='red', s=60, label="优化路径", edgecolor="black")
plt.xlabel("x (优化参数)", fontsize=12)
plt.ylabel("f(x) (损失值)", fontsize=12)
plt.title("梯度下降 - 优化路径 (学习率=0.3)", fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.show()