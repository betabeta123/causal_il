# -*- coding: utf-8 -*-            
# @Time : 2023/12/29 3:04
# @Author: Lily Tian
# @FileName: imitation_learning_plus+.py
# @Software: PyCharm

#此代码用于模型训练和iid的测试

import torch
import os
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use('agg')  # 以 'TkAgg' 为例；如有需要，尝试其他后端
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime


timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

# 读取CSV文件
# csv_file_path = './data_process66.csv'              
csv_file_path = './filtered_dataset.csv'
df = pd.read_csv(csv_file_path)

# 划分训练集和测试集
train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)
# 将测试数据集保存为CSV文件
test_data.to_csv('./test_data1_5lie.csv', index=False)#不保存行索引信息
train_data.to_csv('./train_data1_5lie.csv',index=False)

# 提取特征和标签
X_train = torch.tensor(train_data.iloc[:, :-1].values, dtype=torch.float32)
y_train = torch.tensor(train_data.iloc[:, -1].values - 1, dtype=torch.long)  # 标签从1开始，减1变为从0开始


X_test = torch.tensor(test_data.iloc[:, :-1].values, dtype=torch.float32)
y_test = torch.tensor(test_data.iloc[:, -1].values - 1, dtype=torch.long)

# 定义模型
class ImitationModel(nn.Module):
    def __init__(self, input_size, output_size):
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

num_epochs = 2000
lr=0.0001

# 初始化模型、损失函数和优化器
input_size = X_train.shape[1]
output_size = torch.max(y_train) + 1
model = ImitationModel(input_size, output_size)
print(model)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr) 
# optimizer = optim.Adadelta(model.parameters(), lr)
# 训练模型
train_losses = []
test_losses = []
accuracies = []
max_accuracy = 0.0

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    train_losses.append(loss.item())

    # 在测试集上计算损失和准确率
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        test_loss = criterion(test_outputs, y_test)
        test_losses.append(test_loss.item())

        _, predicted_labels = torch.max(test_outputs, 1)
        accuracy = (predicted_labels == y_test).sum().item() / len(y_test)
        accuracies.append(accuracy)


        #更新最高准确率
        if accuracy > max_accuracy:
            max_accuracy = accuracy

    # 打印每个epoch的损失和准确率
    print(f'Epoch [{epoch+1}/{num_epochs}], '
          f'Train Loss: {loss.item():.4f}, '
          f'Test Loss: {test_loss.item():.4f}, '
          f'Accuracy: {accuracy:.4f}, '
          f'Max_Accuracy: {max_accuracy:.4f}')

save_dir = "./models"
model_save_path=os.path.join(save_dir,f'BC_trained_{timestamp}.pt')
print("Model will be saved at:", model_save_path)
torch.save(model.state_dict(), model_save_path)#将神经网络的参数保存到BC_trained.pt中 


# 可视化损失和准确率
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig(os.path.join(save_dir, f'loss_plot_{timestamp}.png'))

plt.subplot(1, 2, 2)
plt.plot(accuracies, label='Intervention-based Counterfactual Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
# plt.show()
plt.savefig(os.path.join(save_dir, f'accuracy_plot_{timestamp}.png'))