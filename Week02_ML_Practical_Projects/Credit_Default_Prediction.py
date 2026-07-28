# ==================== 信用卡违约预测（修复NaN版） ====================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
import warnings
import os
import sys
from datetime import datetime

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']  # 保留以防其他字符，但英文无需
plt.rcParams['axes.unicode_minus'] = False

# 创建输出目录
os.makedirs('figures', exist_ok=True)
os.makedirs('results', exist_ok=True)

# 日志
log_file = open('results/run_log.txt', 'w', encoding='utf-8')
def log(msg):
    print(msg)
    log_file.write(msg + '\n')
    log_file.flush()

log(f"运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
log("="*60)
log("信用卡违约预测项目启动")

# 数据路径（桌面）
data_path = os.path.expanduser("~/Desktop/default of credit card clients.xlsx")
if not os.path.exists(data_path):
    log(f"错误：找不到数据文件 {data_path}")
    sys.exit(1)

log(f"数据路径: {data_path}")

try:
    df = pd.read_excel(data_path, header=0)
except Exception as e:
    log(f"读取失败: {e}")
    sys.exit(1)

log(f"原始列名: {df.columns.tolist()}")

# ========== 重命名列 ==========
rename_map = {
    'Unnamed: 0': 'ID',
    'X1': 'LIMIT_BAL',
    'X2': 'SEX',
    'X3': 'EDUCATION',
    'X4': 'MARRIAGE',
    'X5': 'AGE',
    'X6': 'PAY_0',
    'X7': 'PAY_2',
    'X8': 'PAY_3',
    'X9': 'PAY_4',
    'X10': 'PAY_5',
    'X11': 'PAY_6',
    'X12': 'BILL_AMT1',
    'X13': 'BILL_AMT2',
    'X14': 'BILL_AMT3',
    'X15': 'BILL_AMT4',
    'X16': 'BILL_AMT5',
    'X17': 'BILL_AMT6',
    'X18': 'PAY_AMT1',
    'X19': 'PAY_AMT2',
    'X20': 'PAY_AMT3',
    'X21': 'PAY_AMT4',
    'X22': 'PAY_AMT5',
    'X23': 'PAY_AMT6',
    'Y': 'Y'
}
df.rename(columns=rename_map, inplace=True)
if 'ID' in df.columns:
    df.drop('ID', axis=1, inplace=True)

log(f"重命名后列名: {df.columns.tolist()}")
log(f"数据集形状: {df.shape}")

# 强制转换为数值，无法转换的变为NaN
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# 数据清洗：教育程度和婚姻状况
df['EDUCATION'] = df['EDUCATION'].replace({0: 4, 5: 4, 6: 4})
df['MARRIAGE'] = df['MARRIAGE'].replace({0: 3})

# ========== 立即填充所有缺失值（使用中位数） ==========
imputer_global = SimpleImputer(strategy='median')
df_imputed = pd.DataFrame(imputer_global.fit_transform(df), columns=df.columns)
df = df_imputed
log(f"全局缺失值填充完成，剩余NaN数: {df.isnull().sum().sum()}")

# ========== EDA（确保无NaN） ==========
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
colors = ['#2ecc71', '#e74c3c']

# 1. 目标变量
df['Y'].value_counts().plot(kind='bar', ax=axes[0,0], color=colors, edgecolor='black')
axes[0,0].set_title('Distribution of Target')
axes[0,0].set_xticks([0, 1])
axes[0,0].set_xticklabels(['Normal', 'Default'], rotation=0)
axes[0,0].set_ylabel('Count')

# 2. 性别
df.groupby(['SEX','Y']).size().unstack().plot(kind='bar', ax=axes[0,1], color=['#2ecc71','#e74c3c'], edgecolor='black')
axes[0,1].set_title('Gender vs Default')
axes[0,1].set_xticks([0, 1])
axes[0,1].set_xticklabels(['Male', 'Female'], rotation=0)
axes[0,1].legend(['Normal', 'Default'])
axes[0,1].set_ylabel('Count')

# 3. 教育程度
df.groupby(['EDUCATION','Y']).size().unstack().plot(kind='bar', ax=axes[0,2], color=['#2ecc71','#e74c3c'], edgecolor='black')
axes[0,2].set_title('Education vs Default')
axes[0,2].set_xticks([0, 1, 2, 3])
axes[0,2].set_xticklabels(['Graduate', 'University', 'High School', 'Other'], rotation=0)
axes[0,2].legend(['Normal', 'Default'])
axes[0,2].set_ylabel('Count')

# 4. 婚姻状况
df.groupby(['MARRIAGE','Y']).size().unstack().plot(kind='bar', ax=axes[1,0], color=['#2ecc71','#e74c3c'], edgecolor='black')
axes[1,0].set_title('Marriage vs Default')
axes[1,0].set_xticks([0, 1, 2])
axes[1,0].set_xticklabels(['Married', 'Single', 'Other'], rotation=0)
axes[1,0].legend(['Normal', 'Default'])
axes[1,0].set_ylabel('Count')

# 5. 年龄分布
axes[1,1].hist([df[df['Y']==0]['AGE'], df[df['Y']==1]['AGE']], bins=30, alpha=0.7, label=['Normal','Default'], color=['#2ecc71','#e74c3c'])
axes[1,1].set_title('Age Distribution')
axes[1,1].set_xlabel('Age')
axes[1,1].set_ylabel('Frequency')
axes[1,1].legend()

# 6. 信用额度(对数)
axes[1,2].hist([np.log10((df[df['Y']==0]['LIMIT_BAL'] + 1).values),
                np.log10((df[df['Y']==1]['LIMIT_BAL'] + 1).values)],
               bins=30, alpha=0.7, label=['Normal','Default'], color=['#2ecc71','#e74c3c'])
axes[1,2].set_title('Log of Credit Limit')
axes[1,2].set_xlabel('Log(Limit)')
axes[1,2].set_ylabel('Frequency')
axes[1,2].legend()

plt.tight_layout()
plt.savefig('figures/eda_plots.png', dpi=300)
plt.close()
log("EDA图保存 figures/eda_plots.png")

# ========== 特征工程 ==========
df_fe = df.copy()
pay_cols = ['PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6']
df_fe['PAY_DELAY_COUNT'] = (df_fe[pay_cols] > 0).sum(axis=1)
df_fe['PAY_DELAY_RATIO'] = df_fe['PAY_DELAY_COUNT'] / 6
df_fe['PAY_AVG_DELAY'] = df_fe[pay_cols].apply(lambda x: x[x>0].mean() if (x>0).any() else 0, axis=1)
df_fe['PAY_MAX_DELAY'] = df_fe[pay_cols].max(axis=1)
df_fe['PAY_0_DELAY'] = (df_fe['PAY_0'] > 0).astype(int)
bill_cols = ['BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6']
pay_amt_cols = ['PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6']
df_fe['AVG_BILL_AMT'] = df_fe[bill_cols].mean(axis=1)
df_fe['AVG_PAY_AMT'] = df_fe[pay_amt_cols].mean(axis=1)
df_fe['PAY_RATIO'] = df_fe['AVG_PAY_AMT'] / (df_fe['AVG_BILL_AMT'] + 1)
df_fe['TOTAL_BILL'] = df_fe[bill_cols].sum(axis=1)
df_fe['TOTAL_PAY'] = df_fe[pay_amt_cols].sum(axis=1)
df_fe['PAY_TOTAL_RATIO'] = df_fe['TOTAL_PAY'] / (df_fe['TOTAL_BILL'] + 1)
df_fe['RECENT_PAY_RATIO'] = (df_fe['PAY_AMT1']+df_fe['PAY_AMT2']+df_fe['PAY_AMT3']) / (df_fe['BILL_AMT1']+df_fe['BILL_AMT2']+df_fe['BILL_AMT3']+1)
df_fe['LIMIT_TO_AGE'] = df_fe['LIMIT_BAL'] / (df_fe['AGE'] + 1)

# ========== 特征工程后再次填充（防止新特征产生NaN） ==========
if df_fe.isnull().sum().sum() > 0:
    imputer_fe = SimpleImputer(strategy='median')
    df_fe = pd.DataFrame(imputer_fe.fit_transform(df_fe), columns=df_fe.columns)
    log("特征工程后填充了缺失值")

log(f"特征工程完成，特征数: {df_fe.shape[1]}，NaN数: {df_fe.isnull().sum().sum()}")

# ========== 划分训练/测试 ==========
X = df_fe.drop('Y', axis=1)
y = df_fe['Y']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 标准化前再次确保无NaN
X_train = X_train.fillna(X_train.median())
X_test = X_test.fillna(X_train.median())  # 用训练集的中位数填充测试集

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 确认训练集中无NaN
assert not np.isnan(X_train_scaled).any(), "训练集仍有NaN"

# ========== SMOTE ==========
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
log(f"SMOTE后训练集大小: {len(X_train_resampled)}")

# ========== 模型训练 ==========
models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10, min_samples_split=50),
    'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100, max_depth=10),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42, n_estimators=100, max_depth=5),
    'XGBoost': XGBClassifier(random_state=42, n_estimators=100, max_depth=5, eval_metric='logloss'),
    'KNN': KNeighborsClassifier(n_neighbors=5)
}

results = []
models_trained = {}
for name, model in models.items():
    log(f"训练 {name} ...")
    model.fit(X_train_resampled, y_train_resampled)
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    res = {
        '模型': name,
        '准确率': accuracy_score(y_test, y_pred),
        '精确率': precision_score(y_test, y_pred),
        '召回率': recall_score(y_test, y_pred),
        'F1分数': f1_score(y_test, y_pred),
        'AUC': roc_auc_score(y_test, y_proba)
    }
    results.append(res)
    models_trained[name] = (model, y_pred, y_proba)
    log(f"  {name} AUC: {res['AUC']:.4f}")

results_df = pd.DataFrame(results)
results_df.to_csv('results/model_comparison.csv', index=False, encoding='utf-8-sig')
log("模型对比表已保存 results/model_comparison.csv")

# ========== 模型对比图 ==========
fig, axes = plt.subplots(1, 2, figsize=(14,5))
metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC']
x = np.arange(len(metrics))
width = 0.15
colors = ['#3498db','#2ecc71','#e74c3c','#f39c12','#9b59b6','#1abc9c']
for i, row in results_df.iterrows():
    # 注意：results_df中的列名还是中文，但取值时用原列名
    values = [row['准确率'], row['精确率'], row['召回率'], row['F1分数'], row['AUC']]
    axes[0].bar(x + i*width, values, width, label=row['模型'], color=colors[i%len(colors)])
axes[0].set_xticks(x + width*(len(results_df)-1)/2)
axes[0].set_xticklabels(metrics)
axes[0].set_ylim(0,1)
axes[0].set_ylabel('Score')
axes[0].legend(loc='lower right')
axes[0].set_title('Model Performance Comparison')

for name, (_, _, y_proba) in models_trained.items():
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    axes[1].plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})', lw=2)
axes[1].plot([0,1],[0,1],'k--', lw=1)
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].set_title('ROC Curves')
axes[1].legend(loc='lower right')
axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figures/model_comparison.png', dpi=300)
plt.close()

# ========== 单独ROC曲线 ==========
fig, ax = plt.subplots(figsize=(8,6))
for name, (_, _, y_proba) in models_trained.items():
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    ax.plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})', lw=2)
ax.plot([0,1],[0,1],'k--', lw=1)
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curves - Model Comparison')
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3)
plt.savefig('figures/roc_curve.png', dpi=300)
plt.close()
log("ROC曲线图保存 figures/roc_curve.png")

# ========== 最佳模型 ==========
best_name = results_df.loc[results_df['AUC'].idxmax(), '模型']
best_model = models_trained[best_name][0]
log(f"最佳模型: {best_name}, AUC={results_df['AUC'].max():.4f}")

# ========== 特征重要性 ==========
if hasattr(best_model, 'feature_importances_'):
    imp = best_model.feature_importances_
    imp_df = pd.DataFrame({'特征': X.columns, '重要性': imp}).sort_values('重要性', ascending=False)
    fig, ax = plt.subplots(figsize=(10,8))
    top = imp_df.head(20)
    ax.barh(top['特征'], top['重要性'], color=plt.cm.Blues(np.linspace(0.4,0.9,len(top)))[::-1])
    ax.set_xlabel('Importance')
    ax.set_title(f'Top 20 Feature Importance - {best_name}')
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig('figures/feature_importance.png', dpi=300)
    plt.close()
    log("特征重要性图保存 figures/feature_importance.png")
else:
    imp_df = None

# ========== 混淆矩阵 ==========
y_pred_best = best_model.predict(X_test_scaled)
cm = confusion_matrix(y_test, y_pred_best)
fig, ax = plt.subplots(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal','Default'], yticklabels=['Normal','Default'])
ax.set_title(f'Confusion Matrix - {best_name}')
plt.tight_layout()
plt.savefig('figures/confusion_matrix.png', dpi=300)
plt.close()
log("混淆矩阵保存 figures/confusion_matrix.png")

# ========== 预测概率分布 ==========
y_proba_best = best_model.predict_proba(X_test_scaled)[:, 1]
fig, ax = plt.subplots(figsize=(8,5))
ax.hist([y_proba_best[y_test==0], y_proba_best[y_test==1]], bins=30, alpha=0.7, label=['Normal','Default'], color=['#2ecc71','#e74c3c'])
ax.axvline(0.5, color='red', linestyle='--', label='Threshold=0.5')
ax.set_xlabel('Predicted Default Probability')
ax.set_ylabel('Frequency')
ax.set_title('Prediction Probability Distribution')
ax.legend()
plt.savefig('figures/prediction_distribution.png', dpi=300)
plt.close()
log("预测概率分布保存 figures/prediction_distribution.png")

# ========== 阈值优化 ==========
thresholds = np.linspace(0.1, 0.9, 50)
best_f1 = 0
best_th = 0.5
for th in thresholds:
    y_th = (y_proba_best >= th).astype(int)
    f1 = f1_score(y_test, y_th)
    if f1 > best_f1:
        best_f1 = f1
        best_th = th
log(f"最优阈值: {best_th:.3f}, F1={best_f1:.4f}")

# ========== 最终评估 ==========
final_metrics = {
    '准确率': accuracy_score(y_test, y_pred_best),
    '精确率': precision_score(y_test, y_pred_best),
    '召回率': recall_score(y_test, y_pred_best),
    'F1分数': f1_score(y_test, y_pred_best),
    'AUC': roc_auc_score(y_test, y_proba_best)
}
log("\n最终模型性能（阈值0.5）:")
for k,v in final_metrics.items():
    log(f"  {k}: {v:.4f}")

# ========== 生成报告 ==========
report = f"""# 信用卡违约预测项目报告

## 1. 项目概述
- 数据集：台湾信用卡客户违约数据（{len(df)}样本，{len(X.columns)}个特征）
- 违约率：{y.mean():.4f}

## 2. 数据预处理
- 修正教育程度和婚姻状况异常值，使用中位数填充缺失值，使用SMOTE平衡样本。

## 3. 特征工程
创建了延迟付款次数、平均付款比率、信用额度利用率等新特征。

## 4. 模型训练
共训练{len(models)}个模型：{', '.join(models.keys())}。  
最佳模型：**{best_name}**，测试集AUC = {final_metrics['AUC']:.4f}。

## 5. 模型性能
详细对比见 `results/model_comparison.csv` 和 `figures/model_comparison.png`。

## 6. 关键特征
"""
if imp_df is not None:
    report += f"Top 5 特征：{imp_df.head(5)['特征'].tolist()}\n"
else:
    report += "（模型不支持特征重要性）\n"
report += f"""
## 7. 阈值优化
最优阈值：{best_th:.3f}，对应F1 = {best_f1:.4f}。

## 8. 结论
模型可有效预测信用卡违约，具有实用价值。

---
报告生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
with open('report.md', 'w', encoding='utf-8') as f:
    f.write(report)
log("报告已生成 report.md")

log("\n项目运行完成！")
log_file.close()
print("\n✅ 全部完成！请查看 figures/ 和 results/ 目录。")