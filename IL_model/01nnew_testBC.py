# -*- coding: utf-8 -*-            
# @Time : 2024/1/9 4:34
# @Author: Lily Tian
# @FileName: 01nnew_testBC.py
# @Software: PyCharm

#此代码用于ood的模型测试
import torch
from torch import nn
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  # 以 'TkAgg' 为例；如有需要，尝试其他后端
import matplotlib.pyplot as plt

# 读取测试数据集
csv_path = "D:/A项目文件夹/imitationproject/CGIL/diff_distrib/sample_biased_dataset.csv"
df = pd.read_csv(csv_path)
X_test = torch.from_numpy(df.iloc[:, :-1].values).to(torch.float32)
y_test = torch.from_numpy(df.iloc[:, -1].values - 1).to(torch.int64)#测试值的label减去1

# 模型测试
class ImitationModel(nn.Module):
    def __init__(self, input_size, output_size):
        # super(ImitationModel, self).__init__()
        # self.fc = nn.Linear(input_size, output_size)
        super(ImitationModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 128)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(128, 256)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(256, 128)
        self.relu4 = nn.ReLU()
        self.fc5 = nn.Linear(128, 64)
        self.relu5 = nn.ReLU()
        self.fc6 = nn.Linear(64, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.fc4(x)
        x = self.relu4(x)
        x = self.fc5(x)
        x = self.relu5(x)
        x = self.fc6(x)
        return x

# class NeuralNetwork(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.model = nn.Sequential(
#             nn.Linear(22, 128),
#             nn.ReLU(),
#             nn.Linear(128, 256),
#             nn.ReLU(),
#             nn.Linear(256, 81),
#             nn.ReLU(),
#             nn.Linear(81, 9),
#         )
#
#     def forward(self, x):
#         pred = self.model(x)
#         return pred

# 创建神经网络的实例
# model = NeuralNetwork()
input_size = X_test.shape[1]
output_size = torch.max(y_test) + 1
model = ImitationModel(input_size, output_size)
# 加载预训练模型的参数
model.load_state_dict(torch.load("D:/A项目文件夹/imitationproject/CGIL/IL_model/logs/useful.pt"))

# 模型测试
model.eval()
with torch.no_grad():
    outputs = model(X_test)
    _, predicted_labels = torch.max(outputs, 1)

# 计算准确率
accuracy = (predicted_labels == y_test).sum().item() / len(y_test)
print(f'Test Accuracy: {accuracy:.4f}')

# 可视化准确率
plt.plot(predicted_labels.eq(y_test).numpy(), label='Correct Predictions')
plt.xlabel('Data Point')
plt.ylabel('Correct Prediction')
plt.legend()
plt.show()