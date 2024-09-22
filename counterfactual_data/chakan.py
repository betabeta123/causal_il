
import pandas as pd

# 用Pandas加载CSV文件，假设文件名为 'your_dataset.csv'
df = pd.read_csv('./AUG_upsample_dataset.csv')

# 获取并打印最后一列的取值
last_column_values = df.iloc[:, -1]
print("最后一列的取值：", last_column_values)
