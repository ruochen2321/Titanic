# -*- coding: utf-8 -*-
"""
泰坦尼克号生存预测简化版
使用均值和众数填充缺失数据
"""

# ==================== 库导入 ====================
import pandas as pd  # 数据处理和分析
import numpy as np  # 数值计算
from sklearn.linear_model import LogisticRegression  # 逻辑回归模型
from sklearn.preprocessing import LabelEncoder  # 标签编码
from sklearn.model_selection import cross_val_score  # 交叉验证

# ==================== 数据加载 ====================
# 定义测试集和训练集文件路径
test_path = 'test.csv'
train_path = 'train.csv'

# 读取原始数据
df = pd.read_csv(train_path)   # 训练集数据
df_test = pd.read_csv(test_path)  # 测试集数据

# ==================== 训练集预处理 ====================
# ----------- 缺失值填充 -----------
# 1. Embarked(登船港口)字段用众数填充
mode_embarked = df['Embarked'].mode()[0]  # 获取出现次数最多的港口
df['Embarked'].fillna(mode_embarked, inplace=True)  # 填充缺失值

# 2. Fare(船票价格)字段用均值填充
mean_fare = df['Fare'].mean()  # 计算所有乘客的票价均值
df['Fare'].fillna(mean_fare, inplace=True)  # 填充缺失值

# 3. Age(年龄)字段用均值填充
mean_age = df['Age'].mean()  # 计算所有乘客的年龄均值
df['Age'].fillna(mean_age, inplace=True)  # 填充缺失值

# ==================== 特征编码 ====================
# 1. Embarked字段独热编码(将分类变量转换为二进制列)
df = pd.get_dummies(df, columns=['Embarked'], prefix=['Embarked'])

# 2. 性别标签编码(将字符串转换为数值)
le = LabelEncoder()  # 创建标签编码器
df['Sex_encoded'] = le.fit_transform(df['Sex'])  # 转换并存储编码结果(female=0, male=1)

# ==================== 最终特征选择 ====================
# 定义用于模型训练的特征列
selected_features = [
    'Pclass',        # 乘客舱位等级(1/2/3等舱)
    'SibSp',         # 同船兄弟姐妹/配偶数量
    'Parch',         # 同船父母/子女数量
    'Fare',          # 船票价格
    'Sex_encoded',   # 性别(0=女,1=男)
    'Embarked_C',    # 登船港口-Cherbourg(独热编码)
    'Embarked_Q',    # 登船港口-Queenstown(独热编码)
    'Embarked_S',    # 登船港口-Southampton(独热编码)
    'Age'            # 年龄
]

# ==================== 模型训练与评估 ====================
# 准备训练数据
X_train = df[selected_features].copy()  # 特征数据
y_train = df['Survived']  # 目标变量(是否幸存)

# 创建逻辑回归模型
model = LogisticRegression(max_iter=10)  # max_iter参数确保模型收敛

# 5折交叉验证评估模型性能
# 每次迭代将数据分为5份，4份训练，1份验证，重复5次
scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
mean_accuracy = scores.mean()  # 计算平均准确率

print(f"模型交叉验证平均准确率: {mean_accuracy:.4f}")

# ==================== 测试集预处理 ====================
# 使用与训练集相同的预处理流程处理测试集

# ----------- 缺失值填充 -----------
# 1. Embarked字段用训练集的众数填充
df_test['Embarked'].fillna(mode_embarked, inplace=True)

# 2. Fare字段用训练集的均值填充
df_test['Fare'].fillna(mean_fare, inplace=True)

# 3. Age字段用训练集的均值填充
df_test['Age'].fillna(mean_age, inplace=True)

# ----------- 特征编码 -----------
# 1. 性别标签编码(使用训练集训练好的编码器)
df_test['Sex_encoded'] = le.transform(df_test['Sex'])  # 注意: 使用transform而不是fit_transform

# 2. Embarked字段独热编码(与训练集相同方式)
df_test = pd.get_dummies(df_test, columns=['Embarked'], prefix=['Embarked'])

# ----------- 特征对齐 -----------
# 确保测试集特征列与训练集一致(处理可能缺失的列)
missing_cols = set(selected_features) - set(df_test.columns)  # 找出训练集有但测试集没有的特征
for col in missing_cols:
    df_test[col] = 0  # 缺失列填充0(因为独热编码可能缺少某些类别)

# 准备测试集特征数据
X_test = df_test[selected_features]

# ==================== 模型训练与预测 ====================
# 使用全部训练数据重新训练模型
model.fit(X_train, y_train)

# 预测测试集生存情况
test_predictions = model.predict(X_test)

# ==================== 结果输出 ====================
# 创建结果DataFrame
output = pd.DataFrame({
    'PassengerId': df_test['PassengerId'],  # 乘客ID
    'Survived': test_predictions.astype(int)  # 生存预测(0=死亡,1=生存)
})

# 保存结果到CSV文件
output.to_csv('titanic_predictions.csv', index=False)
print("预测结果已保存为 titanic_predictions.csv")
print('=== 程序执行完毕 ===')