"""
Iris 鸢尾花分类完整项目
满足要求：
- 读取 Iris 数据
- 输出数据基本信息
- 绘制类别分布图、特征散点图、相关性热力图
- 使用 Logistic Regression、KNN、Decision Tree
- 输出 Accuracy 和混淆矩阵
- 分析哪个特征对分类最有帮助
- 自动保存所有图片到 results/ 目录
- 输出模型指标汇总表
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from datetime import datetime
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# ============================
# 0. 创建结果目录和日志文件
# ============================
os.makedirs('results', exist_ok=True)

# 重定向标准输出到控制台和日志文件
class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

log_file = open('results/run_log.txt', 'w', encoding='utf-8')
log_file.write(f"运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
sys.stdout = Tee(sys.stdout, log_file)

# ============================
# 1. 加载数据
# ============================
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

print("="*50)
print("Iris 数据集信息")
print("="*50)
print(f"特征名称: {feature_names}")
print(f"类别名称: {target_names}")
print(f"数据形状: {X.shape}")
print(f"类别分布: {np.bincount(y)} (每类50个样本)")

# 转换为 DataFrame 便于分析
df = pd.DataFrame(X, columns=feature_names)
df['species'] = y
df['species_name'] = df['species'].map({i: name for i, name in enumerate(target_names)})

print("\n前5行数据:")
print(df.head())

print("\n缺失值统计:")
print(df.isnull().sum())

print("\n统计描述:")
print(df.describe())

# ============================
# 2. 可视化（所有标签改为英文，避免乱码）
# ============================
sns.set_style('whitegrid')

# 2.1 类别分布
plt.figure(figsize=(6, 4))
sns.countplot(x='species_name', data=df, palette='viridis')
plt.title('Iris Species Distribution', fontsize=14)
plt.xlabel('Species')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('results/iris_distribution.png', dpi=150)
plt.close()

# 2.2 特征散点图
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='petal length (cm)', y='petal width (cm)',
                hue='species_name', palette='deep', s=80)
plt.title('Petal Length vs Width (by Species)', fontsize=14)
plt.xlabel('Petal length (cm)')
plt.ylabel('Petal width (cm)')
plt.legend(title='Species')
plt.tight_layout()
plt.savefig('results/iris_feature_plot.png', dpi=150)
plt.close()

# 2.3 相关性热力图
plt.figure(figsize=(8, 6))
corr = df[feature_names].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Iris Feature Correlation Heatmap', fontsize=14)
plt.tight_layout()
plt.savefig('results/iris_correlation.png', dpi=150)
plt.close()

print("\n可视化图片已保存至 results/ 目录")

# ============================
# 3. 建模与评估
# ============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'KNN (k=5)': KNeighborsClassifier(n_neighbors=5),
    'Decision Tree': DecisionTreeClassifier(random_state=42)
}

results = {}
conf_matrices = {}

print("\n" + "="*50)
print("模型评估结果")
print("="*50)

for name, model in models.items():
    if name in ['Logistic Regression', 'KNN (k=5)']:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    results[name] = acc
    conf_matrices[name] = cm
    
    print(f"\n{name}")
    print(f"  准确率: {acc:.4f}")
    print(f"  混淆矩阵:\n{cm}")

# ============================
# 4. 绘制混淆矩阵对比图（标签保持英文）
# ============================
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, (name, cm) in zip(axes, conf_matrices.items()):
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=target_names, yticklabels=target_names)
    ax.set_title(f'{name}\nAccuracy = {results[name]:.4f}')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
plt.tight_layout()
plt.savefig('results/iris_confusion_matrix.png', dpi=150)
plt.close()

# ============================
# 5. 特征重要性分析
# ============================
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X, y)
importances = dt.feature_importances_
feat_imp_dt = pd.Series(importances, index=feature_names).sort_values(ascending=False)

lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_scaled, y_train)
coef_abs_mean = np.abs(lr.coef_).mean(axis=0)
feat_imp_lr = pd.Series(coef_abs_mean, index=feature_names).sort_values(ascending=False)

print("\n" + "="*50)
print("特征重要性分析")
print("="*50)
print("基于决策树:")
print(feat_imp_dt.to_string())
print("\n基于逻辑回归系数（平均绝对值）:")
print(feat_imp_lr.to_string())

most_important_dt = feat_imp_dt.index[0]
most_important_lr = feat_imp_lr.index[0]
print(f"\n结论：")
print(f"  决策树认为最重要的特征: {most_important_dt} (重要性 {feat_imp_dt.iloc[0]:.4f})")
print(f"  逻辑回归认为最重要的特征: {most_important_lr} (系数 {feat_imp_lr.iloc[0]:.4f})")
if most_important_dt == most_important_lr:
    print(f"  两种方法一致认为 '{most_important_dt}' 对分类最有帮助。")
else:
    print(f"  两种方法略有分歧，可结合领域知识判断。")

# ============================
# 6. 保存指标汇总表
# ============================
metrics_df = pd.DataFrame({
    'Model': list(results.keys()),
    'Accuracy': list(results.values())
})
metrics_df.to_csv('results/iris_metrics.csv', index=False)
print("\n模型指标已保存至 results/iris_metrics.csv")

# 关闭日志文件并恢复标准输出
log_file.close()
sys.stdout = sys.__stdout__

print("\n" + "="*50)
print("所有任务完成！")
print("日志已保存到 results/run_log.txt")
print("图片和指标文件在 results/ 目录下")
print("="*50)