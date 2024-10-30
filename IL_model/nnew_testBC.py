# -*- coding: utf-8 -*-            
# @Time : 2024/1/9 4:34
# @Author: Lily Tian
# @FileName: nnew_testBC.py
# @Software: PyCharm

#此代码用于ood的模型测试
import torch
from torch import nn
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  # 以 'TkAgg' 为例；如有需要，尝试其他后端
import matplotlib.pyplot as plt

# 模型测试
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(22, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 81),
            nn.ReLU(),
            nn.Linear(81, 9),
        )

    def forward(self, x):
        pred = self.model(x)
        return pred

# 创建神经网络的实例
model = NeuralNetwork()

# 加载预训练模型的参数
model.load_state_dict(torch.load("D:/A项目文件夹/imitationProject/imitation_learning/models/BC_trained_20240109-043122.pt"))

# 读取测试数据集
csv_path = "D:/A项目文件夹/imitationProject/data/test_data1.csv"
df = pd.read_csv(csv_path)
X_test = torch.from_numpy(df.iloc[:, :-1].values).to(torch.float32)
y_test = torch.from_numpy(df.iloc[:, -1].values - 1).to(torch.int64)#测试值的label减去1

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
