# -*- coding: utf-8 -*-            
# @Time : 2023/12/21 3:31
# @Author: Lily Tian
# @FileName: exp_tansformed.py
# @Software: PyCharm

#转换成指数分布

import pandas as pd
import numpy as np

# 读取数据集
dataset = pd.read_csv("D:/A项目文件夹/imitationProject/data/test_data1.csv")

# 提取前22列的数据
data_columns = dataset.columns[:-1]
data = dataset[data_columns]

# 对数据进行指数变换
data_exp = np.exp(data)

# 将指数变换后的数据和原始标签合并
transformed_dataset = pd.concat([data_exp, dataset.iloc[:, -1]], axis=1)

# 保存处理后的数据集为新的CSV文件
transformed_dataset.to_csv("transformed_dataset.csv", index=False)


num_rows, num_columns = transformed_dataset.shape
print(f"Number of rows: {num_rows}")
print(f"Number of columns: {num_columns}")