# -*- coding: utf-8 -*-            
# @Time : 2023/12/28 15:11
# @Author: Lily Tian
# @FileName: saved_feature_dataset.py
# @Software: PyCharm

import pandas as pd
# 读取CSV文件
file_path = 'D:/A项目文件夹/imitationProject/data/selected_dataset_causal_initial.csv'  # 这边只有12列
df = pd.read_csv(file_path)
# 保留指定的列
#selected_columns = [1, 2, 4, 5, 7, 8, 9, 11, 14, 18, 21,23]
selected_columns = [17,13,7,8,10,4,3,1]
# 17、13、7、8、10、4、3、1  索引后的top8数据列

#订正：取特征【21、9、18、1、11、14、4、5】+label22【从1开始计数】


df_selected = df.iloc[:, selected_columns]

# 保存结果到新的CSV文件
output_path = 'D:/A项目文件夹/imitationProject/data/saved_feature_dataset.csv'  # 替换成你想保存的文件路径
df_selected.to_csv(output_path, index=False,header=None) #不保存行索引和列索引

# 打印输出结果（可选）
print(df_selected)

