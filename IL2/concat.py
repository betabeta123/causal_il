import pandas as pd

# 读取两个CSV文件，假设文件名为 'dataset1.csv' 和 'dataset2.csv'
df1 = pd.read_csv('../IL1/filtered_dataset.csv',skiprows=1)
df2 = pd.read_csv('./filtered_dataset.csv',skiprows=1)

# 使用concat函数将两个数据集合并，axis=0表示按行合并
merged_df = pd.concat([df1, df2], axis=0)

# 打印合并后的数据集
print("合并后的数据集：")
print(merged_df)

# 如果需要保存合并后的数据集到一个新的CSV文件，可以使用to_csv方法
merged_df.to_csv('concat.csv', index=False)
