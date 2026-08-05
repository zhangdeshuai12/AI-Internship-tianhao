# 特征工程说明文档

本文档详细记录了在 Kaggle House Prices 竞赛项目中所实施的全部特征工程步骤。

---

## 1. 缺失值处理

### 1.1 类别特征（填充 `'None'`）

以下特征若缺失，表示该房屋不具备该属性，统一填充为 `'None'`：

- `Alley` – 小巷类型
- `BsmtQual` – 地下室净高
- `BsmtCond` – 地下室一般状况
- `BsmtExposure` – 地下室采光/花园墙
- `BsmtFinType1` – 地下室装修类型（第一类）
- `BsmtFinType2` – 地下室装修类型（第二类）
- `FireplaceQu` – 壁炉质量
- `GarageType` – 车库位置/类型
- `GarageFinish` – 车库内部装修程度
- `GarageQual` – 车库质量
- `GarageCond` – 车库状况
- `PoolQC` – 泳池质量
- `Fence` – 围栏质量
- `MiscFeature` – 其他杂项设施

### 1.2 数值特征（用中位数填充）

数值型特征的缺失值使用训练集该特征的**中位数**填补：

- `LotFrontage` – 临街长度（英尺）
- `GarageYrBlt` – 车库建造年份
- `MasVnrArea` – 砌体饰面面积（平方英尺）

> **理由**：中位数对异常值不敏感，能更好地代表数据中心趋势。

---

## 2. 类别变量编码

全部类别特征采用 **LabelEncoder** 转换为整型数值。

**选择 LabelEncoder 的原因**：
- 类别特征数量多（40+），One-Hot 会导致维度急剧膨胀
- 树模型（XGBoost、LightGBM、CatBoost）对数值型类别标签友好

**编码的完整特征列表**（共 43 个）：

`MSZoning`, `Street`, `Alley`, `LotShape`, `LandContour`, `Utilities`, `LotConfig`, `LandSlope`, `Neighborhood`, `Condition1`, `Condition2`, `BldgType`, `HouseStyle`, `RoofStyle`, `RoofMatl`, `Exterior1st`, `Exterior2nd`, `MasVnrType`, `ExterQual`, `ExterCond`, `Foundation`, `BsmtQual`, `BsmtCond`, `BsmtExposure`, `BsmtFinType1`, `BsmtFinType2`, `Heating`, `HeatingQC`, `CentralAir`, `Electrical`, `KitchenQual`, `Functional`, `FireplaceQu`, `GarageType`, `GarageFinish`, `GarageQual`, `GarageCond`, `PavedDrive`, `PoolQC`, `Fence`, `MiscFeature`, `SaleType`, `SaleCondition`

---

## 3. 新特征构造（基于领域知识）

| 新特征名 | 构造公式 | 业务含义 |
|----------|----------|----------|
| `TotalSF` | `TotalBsmtSF` + `1stFlrSF` + `2ndFlrSF` | 房屋总居住面积（含地下室） |
| `TotalPorchSF` | `OpenPorchSF` + `EnclosedPorch` + `3SsnPorch` + `ScreenPorch` | 所有阳台/门廊面积之和 |
| `TotalBath` | `FullBath` + 0.5×`HalfBath` + `BsmtFullBath` + 0.5×`BsmtHalfBath` | 浴室总数（加权） |
| `HouseAge` | `YrSold` - `YearBuilt` | 房屋在出售时的年龄（年） |
| `YearsSinceRemod` | `YrSold` - `YearRemodAdd` | 自上次翻新以来的年数 |
| `IsRemod` | 1 若 `YearRemodAdd` ≠ `YearBuilt`，否则 0 | 是否经过翻新（二值） |
| `HasGarage` | 1 若 `GarageArea` > 0，否则 0 | 是否有车库（二值） |
| `HasFireplace` | 1 若 `Fireplaces` > 0，否则 0 | 是否有壁炉（二值） |
| `HasCentralAir` | 1 若 `CentralAir` == 'Y'，否则 0 | 是否有中央空调（二值） |

**设计意图**：
- 组合特征能揭示单个属性无法体现的交互效应
- 二值标志明确突出设施的有无，往往比其具体大小更具区分度

---

## 4. 数值特征的对数变换（降低偏态）

对以下严重右偏的特征应用 **log1p** 变换：

```python
skewed_feats = ['LotFrontage', 'LotArea', '1stFlrSF', 'GrLivArea', 'TotalSF', 'TotalPorchSF']
for feat in skewed_feats:
    data[feat] = np.log1p(data[feat].clip(lower=0))