# 实验二 模块1：参数估计（MLE与MAP）- 修复中文乱码
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta 

plt.rcParams['font.sans-serif'] = ['SimHei']  
plt.rcParams['axes.unicode_minus'] = False   

# 1. PyTorch计算MLE和MAP（伯努利分布+Beta先验）
data = torch.tensor([1.,1.,0.,1.,0.])  # 观测样本
p_mle = torch.mean(data)  # MLE：样本均值
alpha = 2  # Beta先验参数α
beta_p = 2 # Beta先验参数β（避免和scipy.beta重名）
p_map = (torch.sum(data)+alpha-1)/(len(data)+alpha+beta_p-2)  # MAP

# 打印结果
print("="*30)
print("参数估计结果")
print("MLE（最大似然估计） =", p_mle.item())
print("MAP（最大后验估计） =", p_map.item())
print("="*30)

# 2. 可视化：似然/先验/后验分布
data_np = np.array([1,1,0,1,0])
N = len(data_np)
sum_x = np.sum(data_np)
p = np.linspace(0,1,100)  # 生成0-1的100个点

# 计算似然、先验、后验
likelihood = p**sum_x * (1-p)**(N-sum_x)
prior = beta.pdf(p, alpha, beta_p)
posterior = p**(sum_x+1) * (1-p)**(N-sum_x+1)  # 后验∝似然×先验
p_mle_np = sum_x / N
p_map_np = (sum_x + alpha -1) / (N + alpha + beta_p -2)

# 绘图
plt.figure(figsize=(8, 6))
plt.plot(p, likelihood, label="Likelihood（似然）", linewidth=2)
plt.plot(p, prior, label="Prior（Beta先验）", linewidth=2)
plt.plot(p, posterior, label="Posterior（后验）", linewidth=2)
plt.axvline(p_mle_np, color='r', linestyle='--', label="MLE", linewidth=2)
plt.axvline(p_map_np, color='g', linestyle='--', label="MAP", linewidth=2)
plt.xlabel("p (伯努利分布参数)", fontsize=12)
plt.ylabel("概率密度/似然值", fontsize=12)
plt.title("MLE vs MAP (伯努利分布+Beta先验)", fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.show()  # 显示图像