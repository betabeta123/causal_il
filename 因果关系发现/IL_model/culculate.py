import pandas as pd

# 读取CSV文件
file_path = "/home/tianlili/data0/CGIL/IL_model/3.csv"  # 替换为你的文件路径
data = pd.read_csv(file_path,header=None)

# 计算每行的均值、方差和最小值
data[8] = data.iloc[:, 1:8].mean(axis=1)  # 第8列：均值
data[9] = data.iloc[:, 1:8].var(axis=1)   # 第9列：方差
data[10] = data.iloc[:, 1:8].min(axis=1)  # 第10列：最小值

# 保存修改后的CSV文件
ood_output_path = "/home/tianlili/data0/CGIL/IL_model/ood_output_file_1.csv"  # 替换为你的输出文件路径
data.to_csv(ood_output_path, index=False)
print("操作完成！结果已保存至", ood_output_path)
