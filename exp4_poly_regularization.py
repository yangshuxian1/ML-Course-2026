import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# =========================
# 全局数据读取与预处理
# =========================
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
X_train = X.iloc[:split_idx].copy()
X_test = X.iloc[split_idx:].copy()
y_train = y.iloc[:split_idx].copy()
y_test = y.iloc[split_idx:].copy()

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

# =========================
# 模块1：数据读取与实验三基线回顾
# =========================
print("=" * 60)
print("模块1：基线线性回归结果")
print("=" * 60)

baseline_model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])
baseline_model.fit(X_train, y_train)
y_pred_baseline = baseline_model.predict(X_test)
mse_baseline = mean_squared_error(y_test, y_pred_baseline)
r2_baseline = r2_score(y_test, y_pred_baseline)

print("Baseline Linear Regression")
print(f"MSE = {mse_baseline:.4f}")
print(f"R²  = {r2_baseline:.4f}")

# =========================
# 模块2：线性基函数回归（多项式回归）
# =========================
print("\n" + "=" * 60)
print("模块2：多项式回归（degree=2）")
print("=" * 60)

poly_model = Pipeline([
    ("preprocessor", preprocessor),
    ("poly", PolynomialFeatures(degree=2)),
    ("regressor", LinearRegression())
])
poly_model.fit(X_train, y_train)
y_pred_poly = poly_model.predict(X_test)
mse_poly = mean_squared_error(y_test, y_pred_poly)
r2_poly = r2_score(y_test, y_pred_poly)

print("Polynomial Regression (degree=2)")
print(f"MSE = {mse_poly:.4f}")
print(f"R²  = {r2_poly:.4f}")

# 绘制训练误差与测试误差曲线
degrees = [1, 2, 3, 4]
train_errors = []
test_errors = []
for d in degrees:
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("poly", PolynomialFeatures(degree=d)),
        ("regressor", LinearRegression())
    ])
    model.fit(X_train, y_train)
    train_errors.append(mean_squared_error(y_train, model.predict(X_train)))
    test_errors.append(mean_squared_error(y_test, model.predict(X_test)))

plt.figure(figsize=(8, 4))
plt.plot(degrees, train_errors, marker='o', label='训练误差', linewidth=2)
plt.plot(degrees, test_errors, marker='s', label='测试误差', linewidth=2)
plt.xlabel("多项式阶数")
plt.ylabel("MSE")
plt.title("训练误差与测试误差曲线")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show(block=True)

# =========================
# 模块3：正则化回归（Ridge / Lasso）
# =========================
print("\n" + "=" * 60)
print("模块3：Ridge & Lasso 正则化")
print("=" * 60)

# Ridge 回归
ridge_model = Pipeline([
    ("preprocessor", preprocessor),
    ("poly", PolynomialFeatures(degree=2)),
    ("regressor", Ridge(alpha=1.0))
])
ridge_model.fit(X_train, y_train)
y_pred_ridge = ridge_model.predict(X_test)
mse_ridge = mean_squared_error(y_test, y_pred_ridge)
r2_ridge = r2_score(y_test, y_pred_ridge)

print("Ridge Regression")
print(f"MSE = {mse_ridge:.4f}")
print(f"R²  = {r2_ridge:.4f}")

# Lasso 回归
lasso_model = Pipeline([
    ("preprocessor", preprocessor),
    ("poly", PolynomialFeatures(degree=2)),
    ("regressor", Lasso(alpha=0.01, max_iter=10000))
])
lasso_model.fit(X_train, y_train)
y_pred_lasso = lasso_model.predict(X_test)
mse_lasso = mean_squared_error(y_test, y_pred_lasso)
r2_lasso = r2_score(y_test, y_pred_lasso)

print("\nLasso Regression")
print(f"MSE = {mse_lasso:.4f}")
print(f"R²  = {r2_lasso:.4f}")

# 绘制正则化强度与测试误差曲线
alphas = [0.001, 0.01, 0.1, 1, 10, 100]
ridge_test_errors = []
for a in alphas:
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("poly", PolynomialFeatures(degree=2)),
        ("regressor", Ridge(alpha=a))
    ])
    model.fit(X_train, y_train)
    ridge_test_errors.append(mean_squared_error(y_test, model.predict(X_test)))

plt.figure(figsize=(8, 4))
plt.plot(alphas, ridge_test_errors, marker='o', linewidth=2)
plt.xscale('log')
plt.xlabel("正则化强度 alpha")
plt.ylabel("测试集 MSE")
plt.title("正则化强度与测试误差曲线")
plt.grid(True)
plt.tight_layout()
plt.show(block=True)

# =========================
# 模块4：模型复杂度与偏差—方差分析
# =========================
# （无代码总结，仅保留核心逻辑，图表已在模块2生成）

# =========================
# 模块5：残差分析与模型对比总结
# =========================
print("\n" + "=" * 60)
print("模块5：残差分析与模型对比")
print("=" * 60)

# 残差图
residuals = y_test - y_pred_ridge
plt.figure(figsize=(8, 4))
plt.scatter(y_pred_ridge, residuals, alpha=0.7)
plt.axhline(0, color='red', linestyle='--', linewidth=2)
plt.xlabel("预测产量")
plt.ylabel("残差")
plt.title("残差图")
plt.grid(True)
plt.tight_layout()
plt.show(block=True)

# 真实值与预测值对比图
plt.figure(figsize=(8, 4))
plt.scatter(y_test, y_pred_ridge, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
plt.xlabel("真实产量")
plt.ylabel("预测产量")
plt.title("真实值与预测值对比图")
plt.grid(True)
plt.tight_layout()
plt.show(block=True)

# 模型结果汇总表
print("\n【模型结果汇总】")
print("-" * 50)
print(f"{'模型':<20} | {'MSE':<10} | {'R²':<10}")
print("-" * 50)
print(f"{'基线线性回归':<20} | {mse_baseline:<10.4f} | {r2_baseline:<10.4f}")
print(f"{'二阶多项式回归':<20} | {mse_poly:<10.4f} | {r2_poly:<10.4f}")
print(f"{'Ridge正则化回归':<20} | {mse_ridge:<10.4f} | {r2_ridge:<10.4f}")
print(f"{'Lasso正则化回归':<20} | {mse_lasso:<10.4f} | {r2_lasso:<10.4f}")
print("-" * 50)