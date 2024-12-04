# -*- coding: utf-8 -*-            
# @Time : 2023/12/21 2:37
# @Author: Lily Tian
# @FileName: uniform_noise.py
# @Software: PyCharm



#均匀噪音，代码运行通过
import pandas as pd
import numpy as np

# 读取数据集
file_path = "D:/A项目文件夹/imitationProject/CGIL/data/data_process66.csv"  # 替换为你的数据集文件路径
df = pd.read_csv(file_path)

# 设置均匀噪音的范围
noise_min = 0.0
noise_max = 0.5

# 获取前 22 列的列名
feature_columns = df.columns[:-1]

# 添加均匀噪音
for column in feature_columns:
    noise = np.random.uniform(low=noise_min, high=noise_max, size=len(df))
    df[column] = df[column] + noise

# 保存修改后的数据集到新的 CSV 文件
output_file_path = "uniform_noisy_dataset.csv"  # 替换为你想保存的文件路径
df.to_csv(output_file_path, index=False)

print(f"Noisy dataset saved to: {output_file_path}")

num_rows, num_columns = df.shape
print(f"Number of rows: {num_rows}")
print(f"Number of columns: {num_columns}")

