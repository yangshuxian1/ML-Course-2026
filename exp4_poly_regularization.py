import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 1. 数据读取 =====================
df = pd.read_csv("smart_agri_tomato_timeseries_raw.csv", parse_dates=["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

X = df.drop(columns=["yield_next_24h", "timestamp"])
y = df["yield_next_24h"]

numeric_features = [
    "temp", "humidity", "light", "co2", "irrigation", "fertilizer_ec",
    "ph", "canopy_temp", "temp_24h_mean", "light_24h_sum",
    "co2_24h_mean", "irrigation_24h_sum", "growth_index", "yield_now"
]
categorical_features = ["greenhouse_id"]

split_idx = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

# ===================== 2. 预处理Pipeline =====================
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])
categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])
preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# ===================== 3. 训练所有模型 =====================
# 1) 基线线性回归
baseline = Pipeline([("pre", preprocessor), ("reg", LinearRegression())])
baseline.fit(X_train, y_train)
y_pred_base = baseline.predict(X_test)
mse_base, r2_base = mean_squared_error(y_test, y_pred_base), r2_score(y_test, y_pred_base)

# 2) 2阶多项式回归
poly = Pipeline([("pre", preprocessor), ("poly", PolynomialFeatures(2)), ("reg", LinearRegression())])
poly.fit(X_train, y_train)
y_pred_poly = poly.predict(X_test)
mse_poly, r2_poly = mean_squared_error(y_test, y_pred_poly), r2_score(y_test, y_pred_poly)

# 3) Ridge回归
ridge = Pipeline([("pre", preprocessor), ("poly", PolynomialFeatures(2)), ("reg", Ridge(alpha=1.0))])
ridge.fit(X_train, y_train)
y_pred_ridge = ridge.predict(X_test)
mse_ridge, r2_ridge = mean_squared_error(y_test, y_pred_ridge), r2_score(y_test, y_pred_ridge)

# 4) Lasso回归
lasso = Pipeline([("pre", preprocessor), ("poly", PolynomialFeatures(2)), ("reg", Lasso(alpha=0.01, max_iter=10000))])
lasso.fit(X_train, y_train)
y_pred_lasso = lasso.predict(X_test)
mse_lasso, r2_lasso = mean_squared_error(y_test, y_pred_lasso), r2_score(y_test, y_pred_lasso)

# ===================== 4. 生成所有图表 =====================
# 图1：模型复杂度-误差曲线（1-4阶多项式）
degrees = [1,2,3,4]
train_err, test_err = [], []
for d in degrees:
    m = Pipeline([("pre", preprocessor), ("poly", PolynomialFeatures(d)), ("reg", LinearRegression())])
    m.fit(X_train, y_train)
    train_err.append(mean_squared_error(y_train, m.predict(X_train)))
    test_err.append(mean_squared_error(y_test, m.predict(X_test)))

plt.figure(figsize=(8,4))
plt.plot(degrees, train_err, marker='o', label='训练误差', linewidth=2)
plt.plot(degrees, test_err, marker='s', label='测试误差', linewidth=2)
plt.xlabel("多项式阶数")
plt.ylabel("MSE")
plt.title("多项式阶数与误差关系")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("exp4_polynomial_degree.png", dpi=300)
plt.show(block=True)

# 图2：Ridge正则化强度-测试误差曲线
alphas = [0.001, 0.01, 0.1, 1, 10, 100]
ridge_err = []
for a in alphas:
    m = Pipeline([("pre", preprocessor), ("poly", PolynomialFeatures(2)), ("reg", Ridge(alpha=a))])
    m.fit(X_train, y_train)
    ridge_err.append(mean_squared_error(y_test, m.predict(X_test)))

plt.figure(figsize=(8,4))
plt.plot(alphas, ridge_err, marker='o', linewidth=2)
plt.xscale('log')
plt.xlabel("正则化强度alpha")
plt.ylabel("测试集MSE")
plt.title("Ridge正则化强度与泛化误差")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("exp4_ridge_alpha.png", dpi=300)
plt.show(block=True)

# 图3：模型对比柱状图（MSE+R²双轴）
models = ["基线线性", "多项式", "Ridge", "Lasso"]
mse_list = [mse_base, mse_poly, mse_ridge, mse_lasso]
r2_list = [r2_base, r2_poly, r2_ridge, r2_lasso]

fig, ax1 = plt.subplots(figsize=(10,5))
ax1.bar(models, mse_list, color='#1f77b4', alpha=0.7, label='MSE')
ax1.set_ylabel("MSE", color='#1f77b4', fontsize=12)
ax1.tick_params(axis='y', labelcolor='#1f77b4')

ax2 = ax1.twinx()
ax2.plot(models, r2_list, color='#ff7f0e', marker='o', linewidth=2, label='R²')
ax2.set_ylabel("R²", color='#ff7f0e', fontsize=12)
ax2.tick_params(axis='y', labelcolor='#ff7f0e')

plt.title("4种模型性能对比")
fig.tight_layout()
plt.savefig("exp4_model_comparison.png", dpi=300)
plt.show(block=True)

# 图4：残差分析合集（3合1大图）
residuals = y_test - y_pred_ridge
fig, axes = plt.subplots(2, 2, figsize=(12,8))

# 子图1：残差散点图
axes[0,0].scatter(y_pred_ridge, residuals, alpha=0.6, color='#2ca02c')
axes[0,0].axhline(0, color='red', linestyle='--')
axes[0,0].set_title("残差散点图")
axes[0,0].set_xlabel("预测值")
axes[0,0].set_ylabel("残差")

# 子图2：残差直方图
axes[0,1].hist(residuals, bins=20, edgecolor='black', alpha=0.7)
axes[0,1].set_title("残差分布直方图")
axes[0,1].set_xlabel("残差")

# 子图3：真实值vs预测值
axes[1,0].scatter(y_test, y_pred_ridge, alpha=0.6, color='#ff7f0e')
axes[1,0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
axes[1,0].set_title("真实值vs预测值")
axes[1,0].set_xlabel("真实值")
axes[1,0].set_ylabel("预测值")

# 隐藏空白子图
axes[1,1].axis('off')
fig.suptitle("Ridge模型残差分析合集", fontsize=14)
plt.tight_layout()
plt.savefig("exp4_residual_analysis.png", dpi=300)
plt.show(block=True)

# 图5：偏差-方差分解曲线（U型）
# 计算偏差、方差、泛化误差
degrees_full = np.arange(1,5)
bias_sq, var, error = [], [], []
for d in degrees_full:
    m = Pipeline([("pre", preprocessor), ("poly", PolynomialFeatures(d)), ("reg", LinearRegression())])
    m.fit(X_train, y_train)
    y_pred_train = m.predict(X_train)
    y_pred_test = m.predict(X_test)
    # 偏差平方 = 训练集MSE
    bias_sq.append(mean_squared_error(y_train, y_pred_train))
    # 方差 = 测试集MSE - 偏差平方
    var.append(mean_squared_error(y_test, y_pred_test) - bias_sq[-1])
    # 泛化误差 = 测试集MSE
    error.append(mean_squared_error(y_test, y_pred_test))

plt.figure(figsize=(8,4))
plt.plot(degrees_full, bias_sq, marker='o', label='偏差平方', linewidth=2)
plt.plot(degrees_full, var, marker='s', label='方差', linewidth=2)
plt.plot(degrees_full, error, marker='^', label='泛化误差', linewidth=2)
plt.xlabel("多项式阶数")
plt.ylabel("误差")
plt.title("偏差-方差-泛化误差权衡")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("exp4_bias_variance.png", dpi=300)
plt.show(block=True)

# ===================== 5. 输出所有结果 =====================
print("\n" + "="*60)
print("【所有模型结果汇总】")
print("="*60)
print(f"基线线性回归: MSE={mse_base:.4f}, R²={r2_base:.4f}")
print(f"2阶多项式回归: MSE={mse_poly:.4f}, R²={r2_poly:.4f}")
print(f"Ridge正则化回归: MSE={mse_ridge:.4f}, R²={r2_ridge:.4f}")
print(f"Lasso正则化回归: MSE={mse_lasso:.4f}, R²={r2_lasso:.4f}")
print("\n✅ 实验四全图+全结果运行完成！")