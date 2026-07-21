"""
California Housing 房价预测完整项目
满足要求：
- 使用 California Housing 数据
- 分析目标变量分布
- 分析特征相关性
- 使用 Linear Regression
- 增加 Ridge 对比
- 输出 MSE、RMSE、MAE、R2
- 绘制真实值 vs 预测值图
- 自动保存所有图片到 results/ 目录
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ============================
# 1. 创建结果目录
# ============================
os.makedirs('results', exist_ok=True)

# 设置绘图风格
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style('whitegrid')

# ============================
# 2. 加载数据
# ============================
print("=" * 60)
print("California Housing 数据集信息")
print("=" * 60)

housing = fetch_california_housing()
X = housing.data
y = housing.target
feature_names = housing.feature_names

print(f"特征名称: {feature_names}")
print(f"数据形状: {X.shape}")
print(f"目标变量描述（前500字符）:\n{housing.DESCR[:500]}...")

# 转换为 DataFrame 便于分析
df = pd.DataFrame(X, columns=feature_names)
df['MedHouseVal'] = y

print("\n前5行数据:")
print(df.head())

print(f"\n缺失值总数: {df.isnull().sum().sum()}")

print("\n统计描述:")
print(df.describe())

# ============================
# 3. 目标变量分布图
# ============================
plt.figure(figsize=(8, 5))
sns.histplot(df['MedHouseVal'], kde=True, bins=50, color='skyblue')
plt.title('California Housing 目标变量分布（房价中位数）', fontsize=14)
plt.xlabel('房价 (单位: $100,000)')
plt.ylabel('频数')
plt.tight_layout()
plt.savefig('results/housing_target_dist.png', dpi=150)
plt.close()
print("\n✅ 已保存: results/housing_target_dist.png")

# ============================
# 4. 特征相关性热力图
# ============================
plt.figure(figsize=(10, 8))
corr = df.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('California Housing 特征相关性热力图', fontsize=14)
plt.tight_layout()
plt.savefig('results/housing_correlation.png', dpi=150)
plt.close()
print("✅ 已保存: results/housing_correlation.png")

# ============================
# 5. 划分数据集并标准化
# ============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================
# 6. 模型训练与评估
# ============================
models = {
    'Linear Regression': LinearRegression(),
    'Ridge (alpha=1.0)': Ridge(alpha=1.0, random_state=42)
}

metrics_list = []

print("\n" + "=" * 60)
print("模型评估指标")
print("=" * 60)

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    metrics_list.append({
        'Model': name,
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2
    })

    print(f"\n{name}")
    print(f"  MSE : {mse:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE : {mae:.4f}")
    print(f"  R²  : {r2:.4f}")

# 保存指标到 CSV
metrics_df = pd.DataFrame(metrics_list)
metrics_df.to_csv('results/metrics.csv', index=False)
print("\n✅ 指标已保存至 results/metrics.csv")
print(metrics_df.to_string(index=False))

# ============================
# 7. 真实值 vs 预测值散点图（双模型对比）
# ============================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, (name, model) in zip(axes, models.items()):
    y_pred = model.predict(X_test_scaled)
    ax.scatter(y_test, y_pred, alpha=0.3, s=10, edgecolor='k')
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
            'r--', lw=2)
    ax.set_xlabel('真实房价')
    ax.set_ylabel('预测房价')
    ax.set_title(f'{name}\nR² = {r2_score(y_test, y_pred):.4f}')
    ax.grid(True)
plt.tight_layout()
plt.savefig('results/regression_pred_vs_true.png', dpi=150)
plt.close()
print("✅ 已保存: results/regression_pred_vs_true.png")

# ============================
# 8. 残差图（额外，增加图表数量）
# ============================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, (name, model) in zip(axes, models.items()):
    y_pred = model.predict(X_test_scaled)
    residuals = y_test - y_pred
    ax.scatter(y_pred, residuals, alpha=0.3, s=10, edgecolor='k')
    ax.axhline(y=0, color='r', linestyle='--', lw=2)
    ax.set_xlabel('预测值')
    ax.set_ylabel('残差')
    ax.set_title(f'{name} 残差图')
    ax.grid(True)
plt.tight_layout()
plt.savefig('results/residual_plots.png', dpi=150)
plt.close()
print("✅ 已保存: results/residual_plots.png")

print("\n" + "=" * 60)
print("所有任务完成！生成文件清单:")
print("  - results/housing_target_dist.png")
print("  - results/housing_correlation.png")
print("  - results/regression_pred_vs_true.png")
print("  - results/residual_plots.png")
print("  - results/metrics.csv")
print("=" * 60)