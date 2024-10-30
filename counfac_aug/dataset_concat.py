# -*- coding: utf-8 -*-            
# @Time : 2024/10/27 16:57
# @Author: Lily Tian
# @FileName: dataset_concat.py
# @Software: PyCharm

# 指定存储数据集的文件夹路径
# folder_path = 'D:/A项目文件夹/imitationProject/CGIL/counfac_aug/Aug_data'  # 替换为实际文件夹路径
import pandas as pd
import glob
import os

# 获取所有CSV文件的路径
file_paths = glob.glob("D:/A项目文件夹/imitationProject/CGIL/counfac_aug/Aug_data/*.csv")
# 使用glob.glob函数获取指定目录下所有以.csv为扩展名的文件路径，并将结果存储在file_paths列表中

print(file_paths)  # 打印出这些文件路径供你检查

# 创建一个空的 DataFrame
df = pd.DataFrame()
# 创建一个空的DataFrame，用于存储合并后的数据

# 逐个读取每个CSV文件，并将其添加到DataFrame中
for file_path in file_paths:
    # 读取CSV文件并添加文件名为一列
    temp_df = pd.read_csv(file_path, encoding='gbk')
    # 使用pd.read_csv函数读取CSV文件，encoding参数指定了文件的编码格式，这里使用GBK编码
    file_name = os.path.basename(file_path)
    print(file_name)
    # 使用os.path.basename函数获取文件名（包含扩展名）
    temp_df['file_name'] = file_name
    # 将文件名作为新的一列添加到temp_df中
    df = df.append(temp_df, ignore_index=True)
    # 使用df.append函数将temp_df合并到主DataFrame df中，ignore_index=True表示重新设置行索引

# 将DataFrame写入新的CSV文件中
df.to_csv("D:/A项目文件夹/imitationProject/CGIL/counfac_aug/output.csv", index=False)
# 使用df.to_csv函数将合并后的数据保存为新的CSV文件，index=False表示不保存行索引