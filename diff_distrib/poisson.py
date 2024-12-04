# -*- coding: utf-8 -*-            
# @Time : 2023/12/21 2:52
# @Author: Lily Tian
# @FileName: poisson.py
# @Software: PyCharm

#加入泊松噪音
import pandas as pd
import numpy as np

# 读取数据集
file_path = "D:/A项目文件夹/imitationProject/CGIL/data/data_process66.csv"  # 替换为你的数据集文件路径
df = pd.read_csv(file_path)

# 提取前22列数据
features = df.iloc[:, :-1]

# 添加泊松噪音
noisy_features = features.apply(lambda x: x + np.random.poisson(size=len(x)))

# 将噪音添加到数据集中
noisy_df = pd.concat([noisy_features, df.iloc[:, -1]], axis=1)

# 保存带有泊松噪音的数据集为 CSV 文件
output_path = "poisson_noisy_dataset.csv"  # 保存文件路径
noisy_df.to_csv(output_path, index=False)

num_rows, num_columns = df.shape
print(f"Number of rows: {num_rows}")
print(f"Number of columns: {num_columns}")