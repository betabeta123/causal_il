
## 筛选类别的文件

import pandas as pd

# 读取CSV文件
csv_file_path =  './top_8_features_and_labels.csv' 
df = pd.read_csv(csv_file_path)

# 选择最后一列中值为5、6、7、9的行
selected_values = [5, 6, 7, 9]
filtered_df = df[df.iloc[:, -1].isin(selected_values)]

# 保存筛选后的数据集
filtered_csv_path = './top_8_features_and_labels_shai.csv'  # 请替换为你想要保存的文件路径
filtered_df.to_csv(filtered_csv_path, index=False)

print(f"Filtered dataset saved to {filtered_csv_path}")