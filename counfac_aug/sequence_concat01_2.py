# 新建一个数据集cc.csv，cc.csv数据集的第[3, 6, 8, 10, 12, 13, 15, 16, 17, 19, 20, 22]列分别取自aa.csv中1~12列的数据，
# 第[1,2,4,5,7,9,11,14,18,21,13]列分别取自数据集bb.csv中的1~11列数据，将新数据集保存至new_data.csv

import pandas as pd
import numpy as np

#新建一个数据集，并将两个数据集合并：合并的数据包括原数据和干预后的数据；
aa_data = pd.read_csv('/home/tianlili/data0/CGIL/counfac_aug/output/noncausal_sequence_withnoise.csv',header=None)#非因果顺序数据
bb_data = pd.read_csv('/home/tianlili/data0/CGIL/counfac_aug/output/AUG_causal_sequence/AUG_causal_sequence_dataset.csv',header=None)#因果顺序数据，
#因果顺序数据需要不断修改数据集
bb_data = bb_data.iloc[1:].reset_index(drop=True)# 因果数据集的列名需要去掉
print(f'aa_data数据集的行列数:{aa_data.shape}')
print(f'bb_data数据集的行列数:{bb_data.shape}')

# 定义 cc.csv 的列数，根据最大列索引确定列数
num_rows = max(aa_data.shape[0], bb_data.shape[0])
num_columns = aa_data.shape[1]+bb_data.shape[1]

# 创建一个空的 DataFrame，预定义行数和列数
new_data = pd.DataFrame(index=range(num_rows), columns=range(num_columns))
# 从 aa.csv 中提取指定列（1~12列），并分别填充到 cc.csv 的指定列
aa_columns_to_copy = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]  # 对应 aa.csv 的第 1~12 列
cc_columns_aa = [2, 5, 7, 9, 11, 12, 14, 15, 16, 18, 19, 21]  # cc.csv 中对应位置的列
for i, cc_col in enumerate(cc_columns_aa):
    new_data.iloc[:, cc_col] = aa_data.iloc[:, aa_columns_to_copy[i]]
# 从 bb.csv 中提取指定列（11列数据），并分别填充到 cc.csv 的剩余列
bb_columns_to_copy = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  
cc_columns_bb = [0, 1, 3, 4, 6, 8, 10, 13, 17, 20, 22]  # cc.csv 中剩余位置的列
for i, cc_col in enumerate(cc_columns_bb):
    new_data.iloc[:, cc_col] = bb_data.iloc[:, bb_columns_to_copy[i]]
new_data.to_csv('/home/tianlili/data0/CGIL/counfac_aug/output/03noncausal_new_concat/03noncausal_cancat_sequence_data_00.csv', header=None,index=False)
print(f'新数据集 new_concat_sequence_data.csv的行数和列数：{new_data.shape}')
print("新数据集已保存到03noncausal_cancat_sequence_data.csv")
