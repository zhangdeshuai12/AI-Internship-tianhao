#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Titanic 完整项目脚本（优化版）
参考：Kaggle Titanic Prediction 0.794 Score
优化点：特征工程增强、超参数调优、Stacking集成
目标分数：0.80+
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import warnings
warnings.filterwarnings('ignore')

# ======================== 配置 ========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = BASE_DIR
OUTPUT_DIR = os.path.join(BASE_DIR, 'output_optimized')
FIGURE_DIR = os.path.join(OUTPUT_DIR, 'figures')
RESULT_DIR = os.path.join(OUTPUT_DIR, 'results')
SCREENSHOT_DIR = os.path.join(OUTPUT_DIR, 'kaggle_submission_screenshots')

for d in [OUTPUT_DIR, FIGURE_DIR, RESULT_DIR, SCREENSHOT_DIR]:
    os.makedirs(d, exist_ok=True)

# ======================== 数据加载 ========================
def load_data():
    train = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
    test = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
    return train, test

# ======================== EDA 图表 ========================
def generate_eda_plots(train):
    """生成 EDA 图表"""
    # 1. 性别生存率
    plt.figure(figsize=(6,4))
    sns.barplot(x='Sex', y='Survived', data=train)
    plt.title('Survival Rate by Sex')
    plt.savefig(os.path.join(FIGURE_DIR, 'titanic_sex_survival.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # 2. 舱位生存率
    plt.figure(figsize=(6,4))
    sns.barplot(x='Pclass', y='Survived', data=train)
    plt.title('Survival Rate by Pclass')
    plt.savefig(os.path.join(FIGURE_DIR, 'titanic_pclass_survival.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # 3. 年龄分布
    plt.figure(figsize=(8,5))
    sns.histplot(train['Age'][train['Survived']==0], color='red', label='Died', kde=True, bins=20)
    sns.histplot(train['Age'][train['Survived']==1], color='green', label='Survived', kde=True, bins=20)
    plt.legend()
    plt.title('Age Distribution by Survival')
    plt.savefig(os.path.join(FIGURE_DIR, 'titanic_age_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # 4. Fare箱线图
    plt.figure(figsize=(8,5))
    sns.boxplot(x='Survived', y='Fare', data=train)
    plt.title('Fare vs Survived')
    plt.savefig(os.path.join(FIGURE_DIR, 'titanic_fare_boxplot.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # 5. 相关性热力图
    num_cols = ['Survived','Pclass','Age','SibSp','Parch','Fare']
    plt.figure(figsize=(8,6))
    sns.heatmap(train[num_cols].corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap')
    plt.savefig(os.path.join(FIGURE_DIR, 'titanic_correlation_heatmap.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # 6. 登船港口生存率
    plt.figure(figsize=(6,4))
    sns.countplot(x='Embarked', hue='Survived', data=train)
    plt.title('Embarked vs Survived')
    plt.savefig(os.path.join(FIGURE_DIR, 'titanic_embarked_survival.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print("EDA 图表已生成并保存至", FIGURE_DIR)

# ======================== 特征工程（增强版） ========================
def feature_engineering(train, test):
    """增强版特征工程"""
    y_train = train['Survived']
    train_ids = train['PassengerId']
    test_ids = test['PassengerId']

    combined = pd.concat([
        train.drop(['Survived', 'PassengerId'], axis=1),
        test.drop('PassengerId', axis=1)
    ], axis=0, ignore_index=True)

    # 1. 提取 Title（称呼）
    combined['Title'] = combined['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())
    
    # 更细致的 Title 分组
    title_mapping = {
        'Mr': 0, 'Miss': 1, 'Mrs': 2, 'Master': 3,
        'Dr': 4, 'Rev': 4, 'Col': 4, 'Major': 4, 'Mlle': 4,
        'Countess': 4, 'Ms': 4, 'Lady': 4, 'Jonkheer': 4,
        'Don': 4, 'Dona': 4, 'Mme': 4, 'Capt': 4, 'Sir': 4
    }
    combined['Title'] = combined['Title'].map(title_mapping).fillna(4).astype(int)

    # 2. 家庭相关特征
    combined['FamilySize'] = combined['SibSp'] + combined['Parch'] + 1
    combined['IsAlone'] = (combined['FamilySize'] == 1).astype(int)
    
    # 3. 家庭规模分组（新增）
    combined['FamilyGroup'] = pd.cut(
        combined['FamilySize'], 
        bins=[0, 1, 3, 6, 100], 
        labels=[0, 1, 2, 3]
    ).astype('float')

    # 4. 船舱特征（新增）
    combined['HasCabin'] = combined['Cabin'].notna().astype(int)
    
    # 5. 提取船舱甲板（新增）
    combined['Deck'] = combined['Cabin'].apply(
        lambda x: x[0] if pd.notna(x) else 'U'
    )
    deck_mapping = {deck: i for i, deck in enumerate(sorted(combined['Deck'].unique()))}
    combined['Deck'] = combined['Deck'].map(deck_mapping).astype(int)

    # 6. 年龄填充（按 Title 分组填充）
    combined['Age'] = combined.groupby('Title')['Age'].transform(
        lambda x: x.fillna(x.median())
    )
    combined['Age'] = combined['Age'].fillna(combined['Age'].median())
    
    # 7. 年龄分段
    combined['AgeBin'] = pd.cut(
        combined['Age'], 
        bins=[0, 12, 20, 35, 60, 100], 
        labels=[0, 1, 2, 3, 4]
    ).astype('float')

    # 8. Fare 处理
    combined['Fare'] = combined['Fare'].fillna(combined['Fare'].median())
    combined['FareBin'] = pd.qcut(
        combined['Fare'], 4, labels=[0, 1, 2, 3], duplicates='drop'
    ).astype('float')
    
    # 9. Fare 对数变换（新增，处理长尾分布）
    combined['FareLog'] = np.log1p(combined['Fare'])

    # 10. Embarked 填充
    combined['Embarked'] = combined['Embarked'].fillna(combined['Embarked'].mode()[0])

    # 11. 删除冗余列
    combined.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

    # 12. 编码分类变量
    combined['Sex'] = combined['Sex'].map({'male': 0, 'female': 1})
    combined['Embarked'] = combined['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

    # 拆分
    X_train = combined.iloc[:len(train)]
    X_test = combined.iloc[len(train):]

    # 强制填充所有剩余 NaN
    print("\n=== 特征工程后缺失值检查 ===")
    print(X_train.isnull().sum())

    for col in X_train.columns:
        if X_train[col].dtype in ['float64', 'int64']:
            median_val = X_train[col].median()
            X_train[col] = X_train[col].fillna(median_val)
            X_test[col] = X_test[col].fillna(median_val)
        else:
            mode_val = X_train[col].mode()[0]
            X_train[col] = X_train[col].fillna(mode_val)
            X_test[col] = X_test[col].fillna(mode_val)

    print("\n=== 填充后缺失值总数 ===")
    print(f"X_train: {X_train.isnull().sum().sum()} (应为0)")
    print(f"X_test: {X_test.isnull().sum().sum()} (应为0)")
    print(f"\n最终特征列表: {list(X_train.columns)}")

    return X_train, X_test, y_train, train_ids, test_ids

# ======================== 模型评估 ========================
def evaluate_models(X, y, models, cv=5):
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    results = []
    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
        mean, std = scores.mean(), scores.std()
        results.append([name, mean, std])
        print(f'{name:25s} CV Accuracy = {mean:.4f} (+/- {std:.4f})')
    return pd.DataFrame(results, columns=['Model', 'CV_Mean', 'CV_Std'])

# ======================== 超参数调优 ========================
def tune_randomforest(X, y):
    """RandomForest 超参数调优"""
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [5, 10, 15],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    rf = RandomForestClassifier(random_state=42)
    grid = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=0)
    grid.fit(X, y)
    print(f"RandomForest 最佳参数: {grid.best_params_}")
    print(f"RandomForest 最佳 CV 分数: {grid.best_score_:.4f}")
    return grid.best_estimator_

def tune_xgboost(X, y):
    """XGBoost 超参数调优"""
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
    xgb = XGBClassifier(random_state=42, eval_metric='logloss', use_label_encoder=False)
    grid = GridSearchCV(xgb, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=0)
    grid.fit(X, y)
    print(f"XGBoost 最佳参数: {grid.best_params_}")
    print(f"XGBoost 最佳 CV 分数: {grid.best_score_:.4f}")
    return grid.best_estimator_

# ======================== 主流程 ========================
def main():
    print("="*60)
    print("Titanic 项目自动运行（优化版）")
    print("目标：0.80+ 分数")
    print("="*60)

    # 1. 加载数据
    train, test = load_data()
    print(f"训练集: {train.shape}, 测试集: {test.shape}")

    # 2. EDA 图表
    generate_eda_plots(train)

    # 3. 特征工程
    X_train, X_test, y_train, train_ids, test_ids = feature_engineering(train, test)

    # 4. 超参数调优（可选，耗时较长，可跳过直接用默认参数）
    print("\n" + "="*60)
    print("开始超参数调优（可能需要几分钟）...")
    print("="*60)
    
    # 调优 RandomForest
    best_rf = tune_randomforest(X_train, y_train)
    
    # 调优 XGBoost
    best_xgb = tune_xgboost(X_train, y_train)
    
    # LightGBM 使用默认参数（也可调优，但为了速度先这样）
    best_lgb = LGBMClassifier(
        n_estimators=200, 
        learning_rate=0.05, 
        max_depth=5,
        random_state=42, 
        verbose=-1
    )

    # 5. 模型对比
    print("\n" + "="*60)
    print("模型交叉验证对比")
    print("="*60)
    
    models = {
        'RandomForest (Tuned)': best_rf,
        'XGBoost (Tuned)': best_xgb,
        'LightGBM': best_lgb,
    }
    cv_df = evaluate_models(X_train, y_train, models)
    cv_df.to_csv(os.path.join(RESULT_DIR, 'titanic_model_comparison.csv'), index=False)
    print(f"\n模型对比表已保存至 {RESULT_DIR}/titanic_model_comparison.csv")

    # 6. Stacking 集成学习
    print("\n" + "="*60)
    print("构建 Stacking 集成模型...")
    print("="*60)
    
    # 基学习器
    base_learners = [
        ('rf', best_rf),
        ('xgb', best_xgb),
        ('lgb', best_lgb)
    ]
    
    # 元学习器（LogisticRegression）
    meta_learner = LogisticRegression(max_iter=1000, random_state=42)
    
    # Stacking 分类器[reference:12][reference:13]
    stacking_clf = StackingClassifier(
        estimators=base_learners,
        final_estimator=meta_learner,
        cv=5,
        stack_method='predict_proba'
    )
    
    # 评估 Stacking 模型
    stacking_scores = cross_val_score(
        stacking_clf, X_train, y_train, 
        cv=5, scoring='accuracy'
    )
    print(f"Stacking CV Accuracy = {stacking_scores.mean():.4f} (+/- {stacking_scores.std():.4f})")
    
    # 添加 Stacking 到对比表
    stacking_row = pd.DataFrame({
        'Model': ['Stacking (RF+XGB+LGB)'],
        'CV_Mean': [stacking_scores.mean()],
        'CV_Std': [stacking_scores.std()]
    })
    cv_df_updated = pd.concat([cv_df, stacking_row], ignore_index=True)
    cv_df_updated.to_csv(os.path.join(RESULT_DIR, 'titanic_model_comparison.csv'), index=False)

    # 7. 训练 Stacking 并生成提交
    print("\n" + "="*60)
    print("训练最终模型并生成提交文件...")
    print("="*60)
    
    stacking_clf.fit(X_train, y_train)
    y_pred = stacking_clf.predict(X_test)
    
    # 第一次提交：Stacking 集成
    sub1 = pd.DataFrame({'PassengerId': test_ids, 'Survived': y_pred})
    sub1_path = os.path.join(RESULT_DIR, 'titanic_submission_v1.csv')
    sub1.to_csv(sub1_path, index=False)
    print(f"第一次提交文件（Stacking）已保存至 {sub1_path}")

    # 第二次提交：单一最优模型（通常是 XGBoost）
    best_xgb.fit(X_train, y_train)
    y_pred_xgb = best_xgb.predict(X_test)
    sub2 = pd.DataFrame({'PassengerId': test_ids, 'Survived': y_pred_xgb})
    sub2_path = os.path.join(RESULT_DIR, 'titanic_submission_v2.csv')
    sub2.to_csv(sub2_path, index=False)
    print(f"第二次提交文件（XGBoost）已保存至 {sub2_path}")

    # 8. 生成运行日志
    log_path = os.path.join(OUTPUT_DIR, 'run_log.txt')
    with open(log_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("Titanic 项目运行日志（优化版）\n")
        f.write("="*60 + "\n")
        f.write(f"训练集样本数: {len(train)}, 测试集样本数: {len(test)}\n")
        f.write(f"特征数量: {X_train.shape[1]}\n")
        f.write("\n模型交叉验证结果:\n")
        f.write(cv_df_updated.to_string(index=False) + "\n")
        f.write("\n提交文件:\n")
        f.write(f"  v1 (Stacking集成): {sub1_path}\n")
        f.write(f"  v2 (XGBoost调优): {sub2_path}\n")
        f.write("\n优化点总结:\n")
        f.write("  1. 新增特征: HasCabin, Deck, FamilyGroup, FareLog\n")
        f.write("  2. RandomForest 和 XGBoost 超参数调优 (GridSearchCV)\n")
        f.write("  3. Stacking 集成 (RF + XGB + LGB → LogisticRegression)\n")
        f.write("\n请手动上传至 Kaggle，并截图保存。\n")
    
    print(f"运行日志已保存至 {log_path}")

    print("\n" + "="*60)
    print("所有任务完成！")
    print("请检查以下目录：")
    print(f"  - 图表: {FIGURE_DIR}")
    print(f"  - 结果文件: {RESULT_DIR}")
    print(f"  - 日志: {log_path}")
    print("\n优化点总结：")
    print("  1. ✅ 新增特征: HasCabin, Deck, FamilyGroup, FareLog")
    print("  2. ✅ 超参数调优: RandomForest + XGBoost (GridSearchCV)")
    print("  3. ✅ Stacking 集成: RF + XGB + LGB → LogisticRegression")
    print("="*60)

if __name__ == '__main__':
    main()