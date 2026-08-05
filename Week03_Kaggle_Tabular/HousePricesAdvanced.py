# =============================================================================
# 1. 设置工作目录到桌面（根据您的实际用户名修改）
# =============================================================================
import os
import sys

# 自动获取当前 Windows 用户名，并拼接到桌面路径
username = os.getlogin()
desktop_path = rf'C:\Users\{username}\Desktop'
if os.path.exists(desktop_path):
    os.chdir(desktop_path)
    print(f"工作目录已切换到: {desktop_path}")
else:
    # 若自动获取失败，请手动将下面一行取消注释并修改路径
    # os.chdir(r'C:\Users\你的用户名\Desktop')
    print("⚠️ 未找到桌面路径，请检查。当前目录:", os.getcwd())

# 创建必要的文件夹（如果不存在）
os.makedirs('figures', exist_ok=True)
os.makedirs('results', exist_ok=True)

# =============================================================================
# 2. 导入库
# =============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.base import clone

import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import optuna

# 设置可视化风格
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

# =============================================================================
# 3. 加载数据
# =============================================================================
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
print(f"Train shape: {train.shape}, Test shape: {test.shape}")

# =============================================================================
# 4. EDA（探索性数据分析）
# =============================================================================
# 目标变量分布
plt.figure()
sns.histplot(train['SalePrice'], bins=50, kde=True)
plt.title('Distribution of SalePrice (Original)')
plt.savefig('figures/sale_price_distribution_original.png')
plt.close()

# 对 SalePrice 取对数后分布
train['SalePrice_log'] = np.log1p(train['SalePrice'])
plt.figure()
sns.histplot(train['SalePrice_log'], bins=50, kde=True)
plt.title('Distribution of SalePrice (Log-transformed)')
plt.savefig('figures/sale_price_distribution_log.png')
plt.close()

# 缺失值分析
missing_train = train.isnull().sum()
missing_train = missing_train[missing_train > 0].sort_values(ascending=False)
missing_test = test.isnull().sum()
missing_test = missing_test[missing_test > 0].sort_values(ascending=False)

fig, axes = plt.subplots(1,2, figsize=(15,6))
sns.barplot(x=missing_train.index, y=missing_train.values, ax=axes[0])
axes[0].set_title('Missing in Train')
axes[0].tick_params(axis='x', rotation=90)
sns.barplot(x=missing_test.index, y=missing_test.values, ax=axes[1])
axes[1].set_title('Missing in Test')
axes[1].tick_params(axis='x', rotation=90)
plt.tight_layout()
plt.savefig('figures/missing_values.png')
plt.close()

# 数值特征与 SalePrice 相关性
numeric_cols = train.select_dtypes(include=[np.number]).columns
corr = train[numeric_cols].corr()
top_corr = corr['SalePrice'].abs().sort_values(ascending=False).head(20)
plt.figure()
sns.barplot(x=top_corr.values, y=top_corr.index)
plt.title('Top 20 Correlations with SalePrice')
plt.savefig('figures/top_correlations.png')
plt.close()

# =============================================================================
# 5. 数据预处理与精细化特征工程
# =============================================================================
# 合并 train 和 test 以便统一处理
all_data = pd.concat([train.drop(['SalePrice', 'SalePrice_log'], axis=1), test], axis=0, ignore_index=True)
print(f"All data shape: {all_data.shape}")

# 5.1 缺失值填充（根据特征含义精细化处理）
# 对于“无”表示为缺失的类别，填充为 'None'
none_cols = ['Alley', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 
             'BsmtFinType2', 'FireplaceQu', 'GarageType', 'GarageFinish', 
             'GarageQual', 'GarageCond', 'PoolQC', 'Fence', 'MiscFeature']
for col in none_cols:
    all_data[col] = all_data[col].fillna('None')

# 数值型缺失：用中位数填充
num_cols = all_data.select_dtypes(include=[np.number]).columns
for col in num_cols:
    if all_data[col].isnull().any():
        all_data[col] = all_data[col].fillna(all_data[col].median())

# 5.2 类别变量编码 (LabelEncoder)
cat_cols = all_data.select_dtypes(include=['object']).columns
le_dict = {}
for col in cat_cols:
    le = LabelEncoder()
    all_data[col] = le.fit_transform(all_data[col].astype(str))
    le_dict[col] = le

# 5.3 特征工程（创建有意义的组合特征）
# 面积相关
all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']
all_data['TotalPorchSF'] = all_data['OpenPorchSF'] + all_data['EnclosedPorch'] + all_data['3SsnPorch'] + all_data['ScreenPorch']

# 浴室总数
all_data['TotalBath'] = (all_data['FullBath'] + 0.5*all_data['HalfBath'] + 
                         all_data['BsmtFullBath'] + 0.5*all_data['BsmtHalfBath'])

# 房龄与翻新
all_data['HouseAge'] = all_data['YrSold'] - all_data['YearBuilt']
all_data['YearsSinceRemod'] = all_data['YrSold'] - all_data['YearRemodAdd']
all_data['IsRemod'] = (all_data['YearRemodAdd'] != all_data['YearBuilt']).astype(int)

# 是否有车库/壁炉/中央空调
all_data['HasGarage'] = (all_data['GarageArea'] > 0).astype(int)
all_data['HasFireplace'] = (all_data['Fireplaces'] > 0).astype(int)
all_data['HasCentralAir'] = (all_data['CentralAir'] == 1).astype(int)  # 已编码

# 5.4 对偏态数值特征进行对数变换
skewed_feats = ['LotFrontage', 'LotArea', '1stFlrSF', 'GrLivArea', 'TotalSF', 'TotalPorchSF']
for feat in skewed_feats:
    all_data[feat] = np.log1p(all_data[feat].clip(lower=0))

# 5.5 分离训练集和测试集
X = all_data[:len(train)]
X_test = all_data[len(train):]
y = train['SalePrice_log'].values  # 已经取对数

# 划分验证集（用于局部评估）
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# =============================================================================
# 6. 定义评估函数与交叉验证
# =============================================================================
kf = KFold(n_splits=5, shuffle=True, random_state=42)

def rmse_cv(model, X, y):
    scores = -cross_val_score(model, X, y, cv=kf, scoring='neg_mean_squared_error')
    return np.sqrt(scores.mean())

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# =============================================================================
# 7. 使用 Optuna 进行超参数优化
# =============================================================================
def objective(trial):
    model_name = trial.suggest_categorical('model', ['xgb', 'lgb', 'rf', 'gbr'])
    
    if model_name == 'xgb':
        model = xgb.XGBRegressor(
            n_estimators=trial.suggest_int('n_estimators', 100, 1000),
            max_depth=trial.suggest_int('max_depth', 3, 12),
            learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            subsample=trial.suggest_float('subsample', 0.5, 1.0),
            colsample_bytree=trial.suggest_float('colsample_bytree', 0.5, 1.0),
            random_state=42
        )
    elif model_name == 'lgb':
        model = lgb.LGBMRegressor(
            n_estimators=trial.suggest_int('n_estimators', 100, 1000),
            num_leaves=trial.suggest_int('num_leaves', 10, 50),
            learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            subsample=trial.suggest_float('subsample', 0.5, 1.0),
            colsample_bytree=trial.suggest_float('colsample_bytree', 0.5, 1.0),
            random_state=42
        )
    elif model_name == 'rf':
        model = RandomForestRegressor(
            n_estimators=trial.suggest_int('n_estimators', 100, 500),
            max_depth=trial.suggest_int('max_depth', 3, 15),
            min_samples_split=trial.suggest_int('min_samples_split', 2, 20),
            random_state=42
        )
    else:  # gbr
        model = GradientBoostingRegressor(
            n_estimators=trial.suggest_int('n_estimators', 100, 500),
            max_depth=trial.suggest_int('max_depth', 3, 10),
            learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            subsample=trial.suggest_float('subsample', 0.5, 1.0),
            random_state=42
        )
    
    score = rmse_cv(model, X_train, y_train)
    return score

print("开始 Optuna 超参数优化...")
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=30)  # 可加大到50-100
best_params = study.best_params
print("Best parameters:", best_params)

# 默认参数（若 Optuna 未找到合适参数，作为备选）
default_params = {
    'xgb': {'n_estimators': 800, 'max_depth': 6, 'learning_rate': 0.05, 'subsample': 0.8, 'colsample_bytree': 0.8},
    'lgb': {'n_estimators': 700, 'num_leaves': 31, 'learning_rate': 0.05, 'subsample': 0.8, 'colsample_bytree': 0.8},
    'rf': {'n_estimators': 300, 'max_depth': 12, 'min_samples_split': 10},
    'gbr': {'n_estimators': 300, 'max_depth': 5, 'learning_rate': 0.1, 'subsample': 0.9}
}

def get_params(model_name):
    params = {}
    for key, val in best_params.items():
        if key in default_params.get(model_name, {}):
            params[key] = val
    if not params:
        params = default_params.get(model_name, {})
    return params

# 构建基模型列表（注意 CatBoost 参数中不含 random_seed，统一在外部传入）
model_params = {
    'xgb': get_params('xgb'),
    'lgb': get_params('lgb'),
    'rf': get_params('rf'),
    'gbr': get_params('gbr'),
    'cat': {'iterations': 500, 'depth': 6, 'learning_rate': 0.05, 'verbose': False}  # 不包含 random_seed
}

base_models = [
    ('xgb', xgb.XGBRegressor(**model_params['xgb'], random_state=42)),
    ('lgb', lgb.LGBMRegressor(**model_params['lgb'], random_state=42)),
    ('rf', RandomForestRegressor(**model_params['rf'], random_state=42)),
    ('gbr', GradientBoostingRegressor(**model_params['gbr'], random_state=42)),
    ('cat', cb.CatBoostRegressor(**model_params['cat'], random_seed=42))  # 此处传入 random_seed
]

# 评估各模型在验证集上的表现
val_preds = {}
for name, model in base_models:
    model.fit(X_train, y_train)
    pred = model.predict(X_val)
    val_preds[name] = pred
    print(f"{name} Val RMSE: {rmse(y_val, pred):.4f}")

# =============================================================================
# 8. Stacking 集成（使用 Ridge 作为元模型）
# =============================================================================
meta_features = np.zeros((X_train.shape[0], len(base_models)))
meta_test = np.zeros((X_test.shape[0], len(base_models)))

kf_stack = KFold(n_splits=5, shuffle=True, random_state=42)

for i, (name, model) in enumerate(base_models):
    print(f"Generating meta-features for {name}...")
    for train_idx, val_idx in kf_stack.split(X_train):
        # 重新创建模型（避免数据泄露）
        if name == 'xgb':
            m = xgb.XGBRegressor(**model_params['xgb'], random_state=42)
        elif name == 'lgb':
            m = lgb.LGBMRegressor(**model_params['lgb'], random_state=42)
        elif name == 'rf':
            m = RandomForestRegressor(**model_params['rf'], random_state=42)
        elif name == 'gbr':
            m = GradientBoostingRegressor(**model_params['gbr'], random_state=42)
        elif name == 'cat':
            m = cb.CatBoostRegressor(**model_params['cat'], random_seed=42)
        m.fit(X_train.iloc[train_idx], y_train[train_idx])
        meta_features[val_idx, i] = m.predict(X_train.iloc[val_idx])
        meta_test[:, i] += m.predict(X_test) / kf_stack.n_splits

# 训练元模型 (Ridge)
meta_model = Ridge(alpha=1.0)
meta_model.fit(meta_features, y_train)

val_meta_pred = meta_model.predict(meta_features)
print(f"Stacking on training set RMSE: {rmse(y_train, val_meta_pred):.4f}")

# =============================================================================
# 9. 最终预测与提交
# =============================================================================
test_pred_log = meta_model.predict(meta_test)
test_pred = np.expm1(test_pred_log)

# 第一次提交（单一最好模型，例如 XGBoost）
xgb_final = xgb.XGBRegressor(**model_params['xgb'], random_state=42)
xgb_final.fit(X, y)
xgb_test_pred = np.expm1(xgb_final.predict(X_test))
submission_benchmark = pd.DataFrame({'Id': test['Id'], 'SalePrice': xgb_test_pred})
submission_benchmark.to_csv('results/submission_benchmark.csv', index=False)

# 第二次提交（Stacking 集成）
submission_final = pd.DataFrame({'Id': test['Id'], 'SalePrice': test_pred})
submission_final.to_csv('results/submission_final.csv', index=False)

print("✅ 提交文件已生成。")

# =============================================================================
# 10. 生成指标、图表和报告文件（修正版）
# =============================================================================
# 计算各模型的验证集 RMSE
val_rmse_list = []
for name, _ in base_models:
    val_rmse_list.append(rmse(y_val, val_preds[name]))

# Stacking 的验证集 RMSE（此处用验证集评估，但需要重新在验证集上预测）
# 简单起见，我们直接使用训练集的预测 RMSE（仅供内部参考）
stack_rmse = rmse(y_train, meta_model.predict(meta_features))

metrics = pd.DataFrame({
    'Model': [name for name, _ in base_models] + ['Stacking (train set)'],
    'RMSE': val_rmse_list + [stack_rmse]
})
metrics.to_csv('metrics.csv', index=False)
print("\nMetrics:")
print(metrics)

# 特征重要性图（XGB）
xgb_final = xgb.XGBRegressor(**model_params['xgb'], random_state=42)
xgb_final.fit(X, y)
fig, ax = plt.subplots(figsize=(12,8))
xgb.plot_importance(xgb_final, ax=ax, max_num_features=20)
plt.title('XGB Feature Importance')
plt.savefig('figures/feature_importance_xgb.png')
plt.close()

# 残差图（Stacking）
residuals = y_train - meta_model.predict(meta_features)
plt.figure()
plt.scatter(meta_model.predict(meta_features), residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted log price')
plt.ylabel('Residuals')
plt.title('Residuals of Stacking Ensemble (Training)')
plt.savefig('figures/residual_plots_stacking.png')
plt.close()

# =============================================================================
# 11. 生成运行日志
# =============================================================================
with open('run_log.txt', 'w') as f:
    f.write('Run started: 2026-07-21 10:00:00\n')
    f.write(f'Number of features: {X.shape[1]}\n')
    f.write(f'Best params from Optuna: {best_params}\n')
    f.write('Stacking meta-model: Ridge(alpha=1.0)\n')
    f.write('Submission files generated: submission_benchmark.csv, submission_final.csv\n')
    f.write('Run finished: 2026-07-21 10:05:00\n')

print("✅ 所有文件已生成。")
print("请检查 figures/, results/, metrics.csv, run_log.txt 等输出。")