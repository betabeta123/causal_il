# -*- coding: utf-8 -*-            
# @Time : 2023/11/27 15:41
# @Author: Lily Tian
# @FileName: Gauss.py
# @Software: PyCharm

#加入高斯噪音，代码运行通过。
import numpy as np
import pandas as pd

# 读取数据集（假设数据集是一个CSV文件）
# dataset = pd.read_csv("D:/A项目文件夹/imitationProject/data/test_data1.csv")
dataset = pd.read_csv("D:/A项目文件夹/imitationProject/CGIL/data/data_process66.csv")
# 提取前22列的数据
data_columns = dataset.columns[:-1]  # 假设最后一列是标签列

# 高斯噪音的均值和方差
mean = 0.01
std_dev = 0.05

# 对前22列添加高斯噪音
noisy_data = dataset[data_columns] + np.random.normal(loc=mean, scale=std_dev, size=dataset[data_columns].shape)

# 将添加噪音后的数据与标签列合并
noisy_dataset = pd.concat([noisy_data, dataset.iloc[:, -1]], axis=1)

# 保存添加噪音后的数据集到新的CSV文件
noisy_dataset.to_csv("Gauss_noisy_dataset_0.01_0.05.csv", index=False)

num_rows, num_columns = dataset.shape

print(f"Number of rows: {num_rows}")
print(f"Number of columns: {num_columns}")