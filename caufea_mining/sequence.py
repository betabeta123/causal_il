
import pandas as pd
# 汉字从1开始计数
# 读取数据集
data = pd.read_csv('/home/tianlili/data0/CGIL/caufea_mining/top_10_features_and_labels.csv')

# 指定列的顺序
columns_order = [
    'feature_1', 'feature_2', 'feature_4', 'feature_5', 
    'feature_7', 'feature_9', 'feature_11', 'feature_14', 
    'feature_18', 'feature_21', 'label'
]

# 按指定顺序重新排列列
data_reordered = data[columns_order]

# 保存新的数据集为 bb.csv
data_reordered.to_csv('/home/tianlili/data0/CGIL/caufea_mining/newsorted_top_10_features_and_labels.csv', index=False)

print("数据集已重新排列并保存为 bb.csv")

#这些特征没有：3、6、8、10、12、13、15、16、17、19、20、22
#索引：
