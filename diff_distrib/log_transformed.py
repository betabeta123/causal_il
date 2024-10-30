# -*- coding: utf-8 -*-            
# @Time : 2023/12/21 3:25
# @Author: Lily Tian
# @FileName: log_transformed.py
# @Software: PyCharm


#数据分布转换成对数分布
import pandas as pd
import numpy as np

# 读取数据集
file_path = "D:/A项目文件夹/imitationProject/data/test_data1.csv"
df = pd.read_csv(file_path)

# 提取前22列数据
features = df.iloc[:, :-1]

# 应用对数转换
log_transformed_features = np.log1p(features)

# 合并转换后的特征和原始标签列
log_transformed_df = pd.concat([log_transformed_features, df.iloc[:, -1]], axis=1)

# 保存处理后的数据集为 CSV 文件
output_path = "log_transformed_dataset.csv"
log_transformed_df.to_csv(output_path, index=False)

num_rows, num_columns = df.shape
print(f"Number of rows: {num_rows}")
print(f"Number of columns: {num_columns}")
