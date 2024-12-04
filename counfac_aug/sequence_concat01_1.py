

#实现从原数据集中抽取非因果数据列，并在非因果数据上加入加性噪音高斯(0,0.05)

import pandas as pd
import numpy as np
#step 8：载入数据，抽取第几列，进行高斯噪音，高斯后的数据要与干预后的因果数据合并。
#读取原始数据集
original_data = pd.read_csv('/home/tianlili/data0/CGIL/data/data_process66.csv', header=None)
print(f'original_data 的行数和列数：{original_data.shape}')
# 选取指定列
columns_to_extract = [3, 6, 8, 10, 12, 13, 15, 16, 17, 19, 20, 22]
df_selected = original_data.iloc[:, [col - 1 for col in columns_to_extract]]  # 减1调整为0索引
# 保存为新的数据集
df_selected.to_csv('/home/tianlili/data0/CGIL/counfac_aug/output/noncausal_sequence.csv',header=None, index=False)
print(f'noncausal_sequence.csv数据集已保存:{df_selected.shape}')
#加入噪音
data01 = pd.read_csv('/home/tianlili/data0/CGIL/counfac_aug/output/noncausal_sequence.csv', header=None)
# 增加高斯噪音（均值为0，标准差为0.05）
noise = np.abs(np.random.normal(0, 0.05, data01.shape))
data_with_noise = data01 + noise
# # 将数据四舍五入到四位小数
# data_with_noise = np.round(data_with_noise, 4)
# # 转换为 DataFrame 如果不是 pandas 格式
# data_with_noise_df = pd.DataFrame(data_with_noise)

# 保存增加噪音后的数据
data_with_noise.to_csv('/home/tianlili/data0/CGIL/counfac_aug/output/noncausal_sequence_withnoise.csv', index=False,header=False,float_format='%.4f')
print(f"noncausal_sequence_withnoise数据已保存:{data_with_noise.shape}")
