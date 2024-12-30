# -*- coding: utf-8 -*-            
# @Time : 2023/12/29 3:04
# @Author: Lily Tian
# @FileName: 01imitation_learning_plus+.py
# @Software: PyCharm


#此代码用于模型训练和iid的测试
#可以看到训练损失和测试损失，以及IID下的准确率，划分成了训练集与测试集
import torch
import os
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
# matplotlib.use('TkAgg')  # 以 'TkAgg' 为例；如有需要，尝试其他后端
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
csv_file_path = '/home/tianlili/data0/CGIL/counfac_aug/AUG_upsample_dataset_all.csv'
df = pd.read_csv(csv_file_path)

# 划分训练集和测试集
train_data, test_data = train_test_split(df, test_size=0.1, random_state=1)
# 将测试数据集保存为CSV文件
test_data.to_csv('/home/tianlili/data0/CGIL/data/test_data2.csv', index=False)#不保存行索引信息


# 提取特征和标签
X_train = torch.tensor(train_data.iloc[:, :-1].values, dtype=torch.float32)
y_train = torch.tensor(train_data.iloc[:, -1].values - 1, dtype=torch.long)  # 标签从1开始，减1变为从0开始

X_test = torch.tensor(test_data.iloc[:, :-1].values, dtype=torch.float32)
y_test = torch.tensor(test_data.iloc[:, -1].values - 1, dtype=torch.long)

# 定义模型
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

num_epochs = 2000
lr=0.01
# 初始化模型、损失函数和优化器
input_size = X_train.shape[1]
output_size = torch.max(y_train) + 1
model = ImitationModel(input_size, output_size)
print(model)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr)

# 训练模型
train_losses = []
test_losses = []
accuracies = []
max_accuracy = 0
max_accuracy_epoch = 0
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

    # 打印每个epoch的损失和准确率
    print(f'Epoch [{epoch+1}/{num_epochs}], '
          f'Train Loss: {loss.item():.4f}, '
          f'Test Loss: {test_loss.item():.4f}, '
          f'Accuracy: {accuracy:.4f}')
    if accuracy > max_accuracy:
        max_accuracy = accuracy
        max_accuracy_epoch = epoch + 1

save_dir = "/home/tianlili/data0/CGIL/IL_model/logs2"
model_save_path=os.path.join(save_dir,f'BC_trained_{timestamp}.pt')
print("Model will be saved at:", model_save_path)
print(f"The maximum accuracy is {max_accuracy:.4f} which occurred at epoch {max_accuracy_epoch}.")
torch.save(model.state_dict(), model_save_path)#将神经网络的参数保存到BC_trained.pt中

# 可视化损失和准确率
# plt.figure(figsize=(12, 4))
# plt.subplot(1, 2, 1)
# plt.plot(train_losses, label='Train Loss')
# plt.plot(test_losses, label='Test Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend()

# plt.subplot(1, 2, 2)
# plt.plot(accuracies, label='Accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.show()
