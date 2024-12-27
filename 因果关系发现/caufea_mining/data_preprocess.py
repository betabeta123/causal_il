# -*- coding: utf-8 -*-            
# @Time : 2023/12/27 17:22
# @Author: Lily Tian
# @FileName: data_preprocess.py
# @Software: PyCharm

#1原始数据处理成初始因果特征集合

import pandas as pd

# 读取CSV文件
file_path = 'D:/A项目文件夹/imitationProject/data/data_process66.csv'  # 替换成你的CSV文件路径
df = pd.read_csv(file_path,header=None)

# 保留指定的列
#selected_columns = [1, 2, 4, 5, 7, 8, 9, 11, 14, 18, 21,23] #因果发现部分的编码是从1开始
selected_columns = [0, 1, 3, 4, 6, 7, 8, 10, 13, 17, 20,22]

df_selected = df.iloc[:, selected_columns]

# 保存结果到新的CSV文件
output_path = 'D:/A项目文件夹/imitationProject/data/selected_dataset_causal_initial.csv'  # 替换成你想保存的文件路径
df_selected.to_csv(output_path, index=False,header=None)

# 打印输出结果（可选）
print(df_selected)


