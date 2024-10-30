# -*- coding: utf-8 -*-            
# @Time : 2023/12/21 3:44
# @Author: Lily Tian
# @FileName: I.py
# @Software: PyCharm
#互信息的计算，可以运行通
#输出三个文件，一个是互信息排列的json文件；另一个是存完的数据top_8_features_and_labels.csv；和互信息可视化图表
#主文件
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
import json
from datetime import datetime
import os
import matplotlib
matplotlib.use('TkAgg')  # 以 'TkAgg' 为例；如有需要，尝试其他后端
import matplotlib.pyplot as plt


# 1. 读取CSV文件
file_path = 'D:/A项目文件夹/imitationProject/data/selected_dataset_causal_initial.csv'
column_names = ['feature_1', 'feature_2', 'feature_4','feature_5', 'feature_7', 'feature_8','feature_9', 'feature_11', 'feature_14','feature_18', 'feature_21','label']#从1开始计数
data = pd.read_csv(file_path,names=column_names)

# 2. 获取前22列的数据和标签列
features = data.iloc[:, :-1]  #选择所有行和除最后一行外的所有列
label = data.iloc[:, -1]  #选择所有行和最后一列


# 3. 计算互信息
mutual_info_values = mutual_info_classif(features, label, random_state=42)

# 4. 将互信息值与特征列对应起来
feature_mi_dict = dict(zip(features.columns, mutual_info_values))
print(feature_mi_dict)

# 5. 按互信息值从大到小进行排序
sorted_feature_mi = sorted(feature_mi_dict.items(), key=lambda x: x[1], reverse=True)

# 6. 打印排序后的互信息值
for feature, mi_value in sorted_feature_mi:
    print(f"{feature}: {mi_value}")

# 7. 取互信息排前8的特征
top_features = [feature for feature, _ in sorted_feature_mi[:10]]

# 8. 提取原始数据中的前10个特征和标签列
selected_data = data[top_features + ['label']]
# 9.保存选定的数据列到csv文件
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
output_csv_path = 'D:/A项目文件夹/imitationProject/CGIL/caufea_mining/'+ timestamp
os.makedirs(output_csv_path, exist_ok=True)
# 拼接保存文件的路径
output_file_path = os.path.join(output_csv_path, 'top_10_features_and_labels.csv')
selected_data.to_csv(output_file_path, index=False, header=True)
print(f'Top 8 features and labels data saved to {output_file_path}')
#8列特征分别是：【从1开始计数】特征21、9、18、1、11、14、4、5、2、7、8

#10、将feature_mi_dict可视化为柱状图
# plt.figure(figsize=(10, 6))
# plt.bar(range(len(feature_mi_dict)), list(feature_mi_dict.values()), align='center')
# plt.xticks(range(len(feature_mi_dict)), list(feature_mi_dict.keys()), rotation=45)
# plt.xlabel('Features')
# plt.ylabel('Mutual Information')
# plt.title('Mutual Information Values for Features')
# plt.tight_layout()
# plt.show()

# 10. 将 feature_mi_dict 可视化为曲线图
plt.figure(figsize=(10, 6))
plt.plot(list(feature_mi_dict.keys()), list(feature_mi_dict.values()), marker='o')
plt.xticks(rotation=45)
plt.xlabel('Features')
plt.ylabel('Mutual Information')
plt.title('Mutual Information Values for Features')
plt.tight_layout()
plt.show()

# 指定保存文件的路径
log_dir ="D:/A项目文件夹/imitationProject/CGIL/caufea_mining/output" + timestamp
os.makedirs(log_dir, exist_ok=True)
# 拼接保存文件的路径
output_file_path_1 = os.path.join(log_dir, 'output.json')

# 使用 json.dumps 将字典转换为 JSON 格式的字符串
json_string = json.dumps(sorted_feature_mi, indent=2)

# 将 JSON 字符串写入文件
with open(output_file_path_1, 'w') as output_file:
    output_file.write(json_string)

print(f'Dictionary saved to {output_file_path_1}')
