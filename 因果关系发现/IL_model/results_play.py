import pandas as pd
import matplotlib.pyplot as plt

# 读取 CSV 文件
csv_file_1 = '/home/tianlili/data0/CGIL/IL_model/logs1/parameter/trainingBC_0.1_results2500_20241218-123002.csv'  
csv_file_2 = '/home/tianlili/data0/CGIL/IL_model/logs1/parameter/trainingBC_0.2_results2500_20241218-124101.csv'  
csv_file_3 = '/home/tianlili/data0/CGIL/IL_model/logs1/parameter/trainingBC_0.3_results2500_20241218-135802.csv'  

csv_file_4 = '/home/tianlili/data0/CGIL/IL_model/logs2/parameter/trainingCGIL_0.1_results2500_20241218-112259_modified_file1.csv'  
csv_file_5 = '/home/tianlili/data0/CGIL/IL_model/logs2/parameter/trainingCGIL_0.2_results2500_20241218-103106_modified_file1.csv' 
csv_file_6 = '/home/tianlili/data0/CGIL/IL_model/logs2/parameter/trainingCGIL_0.3_results2500_20241218-141855_modified_file1.csv' 



# csv_file_1 = '/home/tianlili/data0/CGIL/IL_model/logs1/parameter/trainingBC_0.1_results2500_20241218-123002.csv'  
# csv_file_2 = '/home/tianlili/data0/CGIL/IL_model/logs1/parameter/trainingBC_0.2_results2500_20241218-124101.csv'  
# csv_file_3 = '/home/tianlili/data0/CGIL/IL_model/logs2/parameter/trainingCGIL_0.1_results2500_20241218-112259.csv'  
# csv_file_4 = '/home/tianlili/data0/CGIL/IL_model/logs2/parameter/trainingCGIL_0.2_results2500_20241218-103106.csv' 


df1 = pd.read_csv(csv_file_1)
df2 = pd.read_csv(csv_file_2)
df3 = pd.read_csv(csv_file_3)
df4 = pd.read_csv(csv_file_4)
df5 = pd.read_csv(csv_file_5)
df6 = pd.read_csv(csv_file_6)
window_size = 50  # 例如窗口大小为10，越大平滑效果越明显
# 绘制准确率对比
plt.figure(figsize=(10, 6))

# 使用滚动窗口计算移动平均
df1['Accuracy'] = df1['Accuracy'].rolling(window=window_size).mean()
df2['Accuracy'] = df2['Accuracy'].rolling(window=window_size).mean()
df3['Accuracy'] = df3['Accuracy'].rolling(window=window_size).mean()
df4['Accuracy'] = df4['Accuracy'].rolling(window=window_size).mean()
df5['Accuracy'] = df5['Accuracy'].rolling(window=window_size).mean()
df6['Accuracy'] = df6['Accuracy'].rolling(window=window_size).mean()

# 绘制平滑后的准确率
plt.figure(figsize=(10, 6))
plt.plot(df1['Epoch'], df1['Accuracy'], label='Train:Test=9:1 BC_Accuracy', color='blue')
plt.plot(df2['Epoch'], df2['Accuracy'], label='Train:Test=8:2 BC_Accuracy', color='red')
plt.plot(df3['Epoch'], df3['Accuracy'], label='Train:Test=7:3 BC_Accuracy', color='green')
plt.plot(df4['Epoch'], df4['Accuracy'], label='Train:Test=9:1 CGIL_Accuracy', color='cyan')
plt.plot(df6['Epoch'], df6['Accuracy'], label='Train:Test=8:2 CGIL_Accuracy', color='orange')
plt.plot(df5['Epoch'], df5['Accuracy'], label='Train:Test=7:3 CGIL_Accuracy', color='purple')

# 添加标题和标签
# plt.title('Smoothed Accuracy Comparison Between Two Classifiers')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')

# 显示图例
plt.legend()
plt.grid(True)

# 保存为图片文件
image_save_path = '/home/tianlili/data0/CGIL/IL_model/comparison_plot1223_3.png'
plt.savefig(image_save_path)
plt.close()

print(f"平滑图形已保存为：{image_save_path}")




# # 绘制第一个分类器的准确率
# plt.plot(df1['Epoch'], df1['Accuracy'], label='Classifier 1 Accuracy', color='blue')
# # 绘制第二个分类器的准确率
# plt.plot(df2['Epoch'], df2['Accuracy'], label='Classifier 2 Accuracy', color='red')

# # 添加标题和标签
# plt.title('Accuracy Comparison Between Two Classifiers')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')

# # 显示图例
# plt.legend()

# # 显示图形
# plt.grid(True)
# # plt.show()
# # 保存图形为图片文件
# image_save_path = '/home/tianlili/data0/CGIL/IL_model/comparison_plot.png'  # 设置图片保存路径
# plt.savefig(image_save_path)

# # 打印保存路径
# print(f"图形已保存为：{image_save_path}")

# # 关闭图形（如果不需要进一步使用）
# plt.close()


