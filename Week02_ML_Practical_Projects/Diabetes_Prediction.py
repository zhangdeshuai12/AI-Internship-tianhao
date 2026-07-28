"""
糖尿病预测完整项目脚本
适用数据: diabetes.csv (Pima Indians Diabetes Database)
输出目录: figures/, results/, run_log.txt
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, roc_curve, confusion_matrix)
import warnings
warnings.filterwarnings('ignore')

# =============================================
# 关键修改：自动切换到脚本所在的桌面目录
# =============================================
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
print(f"📂 当前工作目录已切换至: {os.getcwd()}")
print("="*60)

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("⚠️ XGBoost 未安装，将跳过 XGBoost 模型。可运行 pip install xgboost 安装。")

# =============================================
# 1. 环境准备：创建输出目录
# =============================================
os.makedirs('figures', exist_ok=True)
os.makedirs('results', exist_ok=True)
os.makedirs('screenshots', exist_ok=True)

print("🚀 糖尿病预测项目启动")
print("="*60)

# =============================================
# 2. 数据加载与概览
# =============================================
df = pd.read_csv('diabetes.csv')
print(f"\n✅ 数据加载成功，形状: {df.shape}")
print(f"列名: {list(df.columns)}")
print(f"\n目标变量分布:\n{df['Outcome'].value_counts()}")

# =============================================
# 3. 数据清洗（将0值视为缺失，用中位数填充）
# =============================================
zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
print("\n🔧 处理缺失值（将上述特征中的0替换为NaN）...")
df_clean = df.copy()
df_clean[zero_cols] = df_clean[zero_cols].replace(0, np.nan)

missing_counts = df_clean[zero_cols].isnull().sum()
print(f"缺失值统计:\n{missing_counts}")

imputer = SimpleImputer(strategy='median')
df_clean[zero_cols] = imputer.fit_transform(df_clean[zero_cols])
print("✅ 缺失值填充完成（使用中位数）。")

# =============================================
# 4. 探索性数据分析 (EDA) —— 生成至少4张图
# =============================================
print("\n📊 生成 EDA 图表...")

# 图1: 特征分布直方图
df_clean.hist(figsize=(12, 10), bins=20, edgecolor='black')
plt.suptitle('Feature Distributions', y=1.02)          # MODIFIED
plt.tight_layout()
plt.savefig('figures/feature_distributions.png', dpi=300)
plt.close()

# 图2: 相关性热力图
plt.figure(figsize=(10, 8))
sns.heatmap(df_clean.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap')                       # MODIFIED
plt.tight_layout()
plt.savefig('figures/correlation_heatmap.png', dpi=300)
plt.close()

# 图3: 按结果分组的箱线图（展示4个关键特征）
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
key_features = ['Glucose', 'BMI', 'Age', 'DiabetesPedigreeFunction']
for i, col in enumerate(key_features):
    row, col_idx = i // 2, i % 2
    sns.boxplot(x='Outcome', y=col, data=df_clean, ax=axes[row, col_idx])
    axes[row, col_idx].set_title(f'{col} by Outcome')   # MODIFIED
plt.suptitle('Boxplots of Key Features by Outcome')     # MODIFIED
plt.tight_layout()
plt.savefig('figures/boxplots_by_outcome.png', dpi=300)
plt.close()

# 图4: 特征两两散点图矩阵（仅取前6个特征，加速绘图）
sns.pairplot(df_clean, vars=['Glucose', 'BMI', 'Age', 'Insulin'], hue='Outcome', diag_kind='kde')
plt.suptitle('Pairplot of Selected Features', y=1.02)  # MODIFIED
plt.tight_layout()
plt.savefig('figures/pairplot_matrix.png', dpi=300)
plt.close()
print("✅ EDA 图表已保存至 figures/ 目录。")

# =============================================
# 5. 特征与标签分离，划分训练/测试集
# =============================================
X = df_clean.drop('Outcome', axis=1)
y = df_clean['Outcome']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n📌 训练集: {X_train.shape[0]} 条，测试集: {X_test.shape[0]} 条")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =============================================
# 6. 模型训练与评估
# =============================================
print("\n🤖 开始训练模型...")

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(probability=True, random_state=42)
}

if XGB_AVAILABLE:
    models['XGBoost'] = XGBClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.1,
        use_label_encoder=False, eval_metric='logloss', random_state=42
    )

results_list = []
best_auc = 0
best_model_name = ""

for name, model in models.items():
    if name in ['Logistic Regression', 'SVM']:
        X_tr, X_te = X_train_scaled, X_test_scaled
    else:
        X_tr, X_te = X_train, X_test
    
    model.fit(X_tr, y_train)
    y_pred = model.predict(X_te)
    y_proba = model.predict_proba(X_te)[:, 1] if hasattr(model, "predict_proba") else None
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba) if y_proba is not None else None
    
    cv_scores = cross_val_score(model, X_tr, y_train, cv=5, scoring='roc_auc') if hasattr(model, "predict_proba") else [0]
    
    results_list.append({
        'Model': name,
        'Accuracy': round(acc, 4),
        'Precision': round(prec, 4),
        'Recall': round(rec, 4),
        'F1': round(f1, 4),
        'AUC': round(auc, 4) if auc else None,
        'CV_AUC_Mean': round(cv_scores.mean(), 4) if auc else None
    })
    
    if auc and auc > best_auc:
        best_auc = auc
        best_model_name = name
    
    print(f"  ✅ {name} 训练完成，AUC: {auc:.4f}" if auc else f"  ✅ {name} 训练完成")

df_results = pd.DataFrame(results_list)
df_results.to_csv('results/model_comparison.csv', index=False)
print("\n📋 模型对比表已保存至 results/model_comparison.csv")
print(df_results.to_string(index=False))

# =============================================
# 7. 可视化评估：ROC曲线汇总 + 混淆矩阵 + 特征重要性
# =============================================
print("\n📈 生成评估图表...")

# 图5: ROC曲线汇总（包含所有模型）
plt.figure(figsize=(8, 6))
for name, model in models.items():
    if name in ['Logistic Regression', 'SVM']:
        X_te = X_test_scaled
    else:
        X_te = X_test
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_te)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc_val = roc_auc_score(y_test, y_proba)
        plt.plot(fpr, tpr, label=f'{name} (AUC={auc_val:.3f})')

plt.plot([0, 1], [0, 1], 'k--', linewidth=0.8)
plt.xlabel('False Positive Rate')                      # MODIFIED
plt.ylabel('True Positive Rate')                       # MODIFIED
plt.title('ROC Curves Comparison')                     # MODIFIED
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.savefig('results/roc_curve.png', dpi=300)
plt.close()

# 图6: 混淆矩阵（选择最优模型）
best_model = models[best_model_name]
if best_model_name in ['Logistic Regression', 'SVM']:
    X_tr_best, X_te_best = X_train_scaled, X_test_scaled
else:
    X_tr_best, X_te_best = X_train, X_test
best_model.fit(X_tr_best, y_train)
y_pred_best = best_model.predict(X_te_best)

cm = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['0', '1'], yticklabels=['0', '1'])   # MODIFIED
plt.title(f'Confusion Matrix - {best_model_name}')           # MODIFIED
plt.xlabel('Predicted')                                      # MODIFIED
plt.ylabel('Actual')                                         # MODIFIED
plt.tight_layout()
plt.savefig('results/confusion_matrix.png', dpi=300)
plt.close()

# 图7: 特征重要性（使用随机森林）
if 'Random Forest' in models:
    rf_model = models['Random Forest']
    rf_model.fit(X_train, y_train)
    importances = rf_model.feature_importances_
    feat_imp = pd.Series(importances, index=X.columns).sort_values(ascending=False)
    
    plt.figure(figsize=(8, 5))
    feat_imp.plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title('Feature Importance (Random Forest)')        # MODIFIED
    plt.ylabel('Importance')                               # MODIFIED
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('results/feature_importance.png', dpi=300)
    plt.close()

# 图8: 模型预测概率分布（最优模型）
if hasattr(best_model, "predict_proba"):
    y_proba_best = best_model.predict_proba(X_te_best)[:, 1]
    plt.figure(figsize=(8, 5))
    plt.hist(y_proba_best[y_test == 0], bins=30, alpha=0.7, label='Healthy (actual 0)', color='green')
    plt.hist(y_proba_best[y_test == 1], bins=30, alpha=0.7, label='Diabetic (actual 1)', color='red')
    plt.xlabel('Predicted Probability')                    # MODIFIED
    plt.ylabel('Frequency')                                # MODIFIED
    plt.title(f'Probability Distribution - {best_model_name}')  # MODIFIED
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/probability_distribution.png', dpi=300)
    plt.close()

print("✅ 所有评估图表已保存。")

# =============================================
# 8. 生成运行日志 run_log.txt
# =============================================
log_content = f"""
========================================
糖尿病预测项目运行日志
生成时间: {pd.Timestamp.now()}
========================================

[数据信息]
- 原始样本数: {len(df)}
- 特征数: {len(X.columns)}
- 训练集: {len(X_train)} 条
- 测试集: {len(X_test)} 条

[缺失值处理]
- 填充策略: 中位数 (针对 Glucose, BloodPressure, SkinThickness, Insulin, BMI)

[模型评估结果 (测试集)]
{df_results.to_string(index=False)}

[最优模型]
- 名称: {best_model_name}
- 测试集 AUC: {best_auc:.4f}

[生成文件清单]
- figures/feature_distributions.png
- figures/correlation_heatmap.png
- figures/boxplots_by_outcome.png
- figures/pairplot_matrix.png
- results/roc_curve.png
- results/confusion_matrix.png
- results/feature_importance.png
- results/probability_distribution.png
- results/model_comparison.csv

[环境]
- Python 版本: {pd.__version__} (pandas)
- 依赖库: numpy, matplotlib, seaborn, sklearn, xgboost ({'已安装' if XGB_AVAILABLE else '未安装'})

========================================
执行完成！
========================================
"""

with open('run_log.txt', 'w', encoding='utf-8') as f:
    f.write(log_content)

print("\n📝 运行日志已保存至 run_log.txt")
print("\n" + "="*60)
print("🎉 项目全部完成！请检查 results/ 和 figures/ 目录下的输出。")
print("="*60)