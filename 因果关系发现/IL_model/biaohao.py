import pandas as pd

# 读取CSV文件，将文件名替换为实际的文件名
# df = pd.read_csv('/home/tianlili/data0/CGIL/IL_model/logs2/parameter/trainingCGIL_0.1_results2500_20241218-112259.csv')
# df = pd.read_csv('/home/tianlili/data0/CGIL/IL_model/logs2/parameter/trainingCGIL_0.3_results2500_20241218-141855.csv')
df = pd.read_csv('/home/tianlili/data0/CGIL/IL_model/logs2/parameter/trainingCGIL_0.2_results2500_20241218-103106.csv')

# 重新为 'epoch' 列赋值，生成从1到2500的数值序列来替换原列值
df['Epoch'] = range(1, 2500)

# 将修改后的数据保存为新的CSV文件，这里保存为modified_file.csv，你可以按需更改文件名
# df.to_csv('/home/tianlili/data0/CGIL/IL_model/logs2/parameter/trainingCGIL_0.1_results2500_20241218-112259_modified_file1.csv', index=False)
# df.to_csv('/home/tianlili/data0/CGIL/IL_model/logs2/parameter/trainingCGIL_0.3_results2500_20241218-141855_modified_file1.csv', index=False)
df.to_csv('/home/tianlili/data0/CGIL/IL_model/logs2/parameter/trainingCGIL_0.2_results2500_20241218-103106_modified_file1.csv')
