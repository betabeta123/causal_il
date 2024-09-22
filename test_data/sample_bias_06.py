# -*- coding: utf-8 -*-            
# @Time : 2023/12/21 2:56
# @Author: Lily Tian
# @FileName: sample_bias.py
# @Software: PyCharm

#引入样本偏差，将每列的值乘以（1+bias_factor）
import pandas as pd
import numpy as np

# 读取数据集
file_path = "../IL1/test_data1_5lie.csv"  # 替换为你的数据集文件路径
df = pd.read_csv(file_path)

# 提取前22列数据
features = df.iloc[:, :-1]

# 添加样本偏差
bias_factor = 0.1  # 调整这个值以设置样本偏差的大小
biased_features = features.apply(lambda x: x * (1 + bias_factor))

# 将带有样本偏差的数据添加到数据集中
biased_df = pd.concat([biased_features, df.iloc[:, -1]], axis=1)

# 保存带有样本偏差的数据集为 CSV 文件
output_path = "./biased_06.csv"  # 保存文件路径
biased_df.to_csv(output_path, index=False)

num_rows, num_columns = df.shape
print(f"Number of rows: {num_rows}")
print(f"Number of columns: {num_columns}")