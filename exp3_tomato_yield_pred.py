# ======================================
# 实验三 番茄产量预测 全模块完整代码
# 环境：ml_env | 数据集：smart_agri_tomato_timeseries_raw.csv
# ======================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')  # 屏蔽无关警告
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 解决中文乱码
plt.rcParams['axes.unicode_minus'] = False

# ===================== 模块1：数据读取与基础检查 =====================
print("="*50)
print("模块1：数据读取与基础检查")
print("="*50)
# 读取数据集（确保csv文件和py文件同路径）
df = pd.read_csv('smart_agri_tomato_timeseries_raw.xls')
# 1. 查看数据行列数
rows, cols = df.shape
print(f"1) 数据集规模：{rows} 行，{cols} 列")
# 2. 查看缺失值字段（仅显示有缺失的字段）
missing_info = df.isnull().sum()
missing_cols = missing_info[missing_info > 0]
print("2) 存在缺失值的字段：")
print(missing_cols)
# 3. 标注预测目标
print("3) 预测目标字段：yield_next_24h（未来24小时产量）")
# 查看数据基本信息
print("\n数据前5行预览：")
print(df.head())
print("数据字段类型：")
print(df.dtypes)

# ===================== 模块2：EDA（缺失/分布/相关性/可视化） =====================
print("\n" + "="*50)
print("模块2：EDA探索性数据分析")
print("="*50)
# 2.1 缺失值可视化
missing_counts = df.isnull().sum()
plt.figure(figsize=(8, 4))
missing_counts.plot(kind="bar", color='#1f77b4')
plt.title("缺失值数量统计")
plt.ylabel("缺失值个数")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()

# 2.2 数值特征分布直方图
numeric_cols = [
    "temp","humidity","light","co2","irrigation","fertilizer_ec",
    "ph","canopy_temp","temp_24h_mean","light_24h_sum",
    "co2_24h_mean","irrigation_24h_sum","growth_index",
    "yield_now","yield_next_24h"
]
plt.figure(figsize=(14, 10))
df[numeric_cols].hist(bins=20, color='#ff7f0e', alpha=0.7)
plt.suptitle("数值特征分布直方图", fontsize=16)
plt.tight_layout()
plt.show()

# 2.3 特征相关性矩阵热力图
corr = df[numeric_cols].corr()
plt.figure(figsize=(10, 8))
im = plt.imshow(corr, cmap="coolwarm", interpolation="nearest")
plt.colorbar(im, label="相关系数")
plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha='right')
plt.yticks(range(len(corr.columns)), corr.columns)
plt.title("特征相关性矩阵热力图", fontsize=14)
plt.tight_layout()
plt.show()

# 2.4 关键变量与产量的散点图
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes[0,0].scatter(df["temp"], df["yield_next_24h"], alpha=0.5, color='#2ca02c')
axes[0,0].set_title("温度 vs 未来24小时产量")
axes[0,0].set_xlabel("温度")
axes[0,0].set_ylabel("产量")

axes[0,1].scatter(df["light"], df["yield_next_24h"], alpha=0.5, color='#d62728')
axes[0,1].set_title("光照强度 vs 未来24小时产量")
axes[0,1].set_xlabel("光照强度")
axes[0,1].set_ylabel("产量")

axes[1,0].scatter(df["co2"], df["yield_next_24h"], alpha=0.5, color='#9467bd')
axes[1,0].set_title("CO2浓度 vs 未来24小时产量")
axes[1,0].set_xlabel("CO2浓度")
axes[1,0].set_ylabel("产量")

axes[1,1].scatter(df["growth_index"], df["yield_next_24h"], alpha=0.5, color='#8c564b')
axes[1,1].set_title("生长指数 vs 未来24小时产量")
axes[1,1].set_xlabel("生长指数")
axes[1,1].set_ylabel("产量")
plt.tight_layout()
plt.show()

# 2.5 温室1的时序特征可视化（温度/湿度）
gh1 = df[df["greenhouse_id"] == 1].sort_values("timestamp")
plt.figure(figsize=(12, 4))
plt.plot(gh1["timestamp"][:200], gh1["temp"][:200], label="温度", color='#ff7f0e')
plt.plot(gh1["timestamp"][:200], gh1["humidity"][:200], label="湿度", color='#1f77b4')
plt.legend()
plt.title("温室1 温度/湿度时序变化（前200条）")
plt.xlabel("时间戳")
plt.ylabel("数值")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 输出相关性TOP5（与yield_next_24h）
yield_corr = corr['yield_next_24h'].sort_values(ascending=False)
print("与未来24小时产量相关性TOP5的特征：")
print(yield_corr[1:6])  # 排除自身

# ===================== 模块3：预处理与Pipeline构建 =====================
print("\n" + "="*50)
print("模块3：数据预处理与Pipeline构建")
print("="*50)
# 划分特征与标签
X = df.drop(columns=["yield_next_24h", "timestamp"])  # 特征：剔除目标和时间戳
y = df["yield_next_24h"]  # 标签：预测目标
# 划分数值特征和类别特征
numeric_features = [
    "temp","humidity","light","co2","irrigation","fertilizer_ec",
    "ph","canopy_temp","temp_24h_mean","light_24h_sum",
    "co2_24h_mean","irrigation_24h_sum","growth_index","yield_now"
]
categorical_features = ["greenhouse_id"]  # 类别特征：温室编号

# 构建数值特征处理器：缺失值填充（均值）+ 标准化
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),  # 均值填充缺失值
    ("scaler", StandardScaler())  # 标准化消除量纲
])
# 构建类别特征处理器：缺失值填充（众数）+ 独热编码
categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),  # 众数填充缺失值
    ("onehot", OneHotEncoder(handle_unknown="ignore"))  # 独热编码，忽略未知类别
])
# 合并处理器：按特征类型分别处理
preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),  # 处理数值特征
    ("cat", categorical_transformer, categorical_features)  # 处理类别特征
])
print("Pipeline预处理流程构建完成！")
print(f"数值特征数量：{len(numeric_features)}")
print(f"类别特征数量：{len(categorical_features)}")

# ===================== 模块4：防数据泄漏（时间序列切分） =====================
print("\n" + "="*50)
print("模块4：防数据泄漏 - 时间序列数据切分")
print("="*50)
# 按时间戳排序（核心：时间序列不能打乱）
df = df.sort_values("timestamp").reset_index(drop=True)
# 重新划分特征和标签（基于排序后的数据）
X = df.drop(columns=["yield_next_24h", "timestamp"])
y = df["yield_next_24h"]
# 按8:2切分训练集/测试集（时间顺序：前80%训练，后20%测试）
split_idx = int(len(df) * 0.8)
X_train = X.iloc[:split_idx].copy()
X_test = X.iloc[split_idx:].copy()
y_train = y.iloc[:split_idx].copy()
y_test = y.iloc[split_idx:].copy()
# 仅在训练集上拟合预处理流程，测试集仅转换（核心：防止数据泄漏）
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)
# 输出切分后规模
print(f"训练集特征形状：{X_train_processed.shape}")
print(f"测试集特征形状：{X_test_processed.shape}")
print(f"训练集标签数量：{len(y_train)}")
print(f"测试集标签数量：{len(y_test)}")
print("数据切分完成！全程避免未来信息泄漏到训练集")

# ===================== 模块5：线性回归（手写梯度下降GD） =====================
print("\n" + "="*50)
print("模块5：手写梯度下降实现线性回归")
print("="*50)
# 格式转换：适配矩阵运算（处理稀疏矩阵）
X_train_gd = X_train_processed.toarray() if hasattr(X_train_processed, "toarray") else X_train_processed
X_test_gd = X_test_processed.toarray() if hasattr(X_test_processed, "toarray") else X_test_processed
# 增加偏置项（常数项b，对应X0=1）
X_train_b = np.c_[np.ones((X_train_gd.shape[0], 1)), X_train_gd]
X_test_b = np.c_[np.ones((X_test_gd.shape[0], 1)), X_test_gd]
# 标签格式转换：转为列向量
y_train_np = y_train.values.reshape(-1, 1)
y_test_np = y_test.values.reshape(-1, 1)

# 初始化参数：theta包含权重w和偏置b，全0初始化
theta = np.zeros((X_train_b.shape[1], 1))
lr = 0.01  # 学习率（调优后，确保收敛）
epochs = 500  # 迭代次数
loss_history = []  # 记录每次迭代的损失值

# 梯度下降核心循环
for epoch in range(epochs):
    y_pred = X_train_b @ theta  # 预测值：矩阵乘法
    error = y_pred - y_train_np  # 预测误差
    loss = np.mean(error ** 2)  # 计算MSE损失
    loss_history.append(loss)
    # 计算梯度：MSE的梯度公式
    grad = (2 / X_train_b.shape[0]) * X_train_b.T @ error
    # 更新参数：沿负梯度方向
    theta = theta - lr * grad

# 绘制损失曲线
plt.figure(figsize=(8, 4))
plt.plot(loss_history, color='#1f77b4', linewidth=2)
plt.title("手写梯度下降 - 损失曲线（MSE）")
plt.xlabel("迭代次数（Epoch）")
plt.ylabel("均方误差（MSE）")
plt.grid(alpha=0.3)
plt.show()

# 测试集预测与评估
y_test_pred_gd = X_test_b @ theta
mse_gd = mean_squared_error(y_test_np, y_test_pred_gd)
r2_gd = r2_score(y_test_np, y_test_pred_gd)
# 输出评估指标
print("手写梯度下降线性回归 - 测试集评估结果：")
print(f"均方误差（MSE）：{mse_gd:.4f}")
print(f"决定系数（R²）：{r2_gd:.4f}")
print(f"模型参数数量（含偏置）：{len(theta)}")

# ===================== 模块6：sklearn线性回归（对照模型） =====================
print("\n" + "="*50)
print("模块6：sklearn线性回归（对照模型）")
print("="*50)
# 构建完整Pipeline：预处理 + 线性回归
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())  # sklearn内置线性回归
])
# 训练模型（自动基于训练集做预处理）
model.fit(X_train, y_train)
# 测试集预测
y_test_pred_sklearn = model.predict(X_test)
# 评估模型
mse_sklearn = mean_squared_error(y_test, y_test_pred_sklearn)
r2_sklearn = r2_score(y_test, y_test_pred_sklearn)
# 输出评估指标
print("sklearn线性回归 - 测试集评估结果：")
print(f"均方误差（MSE）：{mse_sklearn:.4f}")
print(f"决定系数（R²）：{r2_sklearn:.4f}")
# 对比手写GD与sklearn结果
print("\n手写GD vs sklearn 结果对比：")
print(f"MSE差值：{abs(mse_gd - mse_sklearn):.4f}")
print(f"R²差值：{abs(r2_gd - r2_sklearn):.4f}")

# ===================== 模块7：残差分析（模型诊断） =====================
print("\n" + "="*50)
print("模块7：残差分析（模型诊断）")
print("="*50)
# 计算残差：真实值 - 预测值（基于sklearn模型，结果更稳定）
residuals = y_test - y_test_pred_sklearn

# 残差散点图（预测值 vs 残差）
plt.figure(figsize=(8, 4))
plt.scatter(y_test_pred_sklearn, residuals, alpha=0.7, color='#ff7f0e')
plt.axhline(0, color="red", linestyle="--", linewidth=2, label="残差=0")
plt.title("残差散点图（预测值 vs 残差）")
plt.xlabel("预测产量")
plt.ylabel("残差")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# 残差分布直方图
plt.figure(figsize=(8, 4))
plt.hist(residuals, bins=20, edgecolor="black", alpha=0.7, color='#2ca02c')
plt.title("残差分布直方图")
plt.xlabel("残差")
plt.ylabel("频次")
plt.grid(alpha=0.3)
plt.show()

# 真实值 vs 预测值散点图
plt.figure(figsize=(8, 4))
plt.scatter(y_test, y_test_pred_sklearn, alpha=0.7, color='#d62728')
# 绘制理想预测线（y=x）
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", linewidth=2, label="理想预测线")
plt.title("真实产量 vs 预测产量")
plt.xlabel("真实产量")
plt.ylabel("预测产量")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# 残差统计信息
print("残差统计信息：")
print(f"残差均值：{np.mean(residuals):.4f}")
print(f"残差标准差：{np.std(residuals):.4f}")
print(f"残差最小值：{np.min(residuals):.4f}")
print(f"残差最大值：{np.max(residuals):.4f}")

print("\n" + "="*50)
print("实验三 所有模块运行完成！")
print("="*50)