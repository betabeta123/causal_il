# -*- coding: utf-8 -*-            
# @Time : 2023/12/22 3:26
# @Author: Lily Tian
# @FileName: Couterfactual.py
# @Software: PyCharm

#该部分的主文件
# 已知有一个23列的csv数据集causal.csv，前22列是因果特征，最后一列是label。


# （0）此处用的是保留下来的因果数据集
# （1）建立神经网络NN，用causal.csv数据集对NN进行训练，用于拟合因果特征下的函数关系，NN的输出为label的预测值label_pre，可视化损失函数曲线和准确率曲线，使得网络训练良好，训练完成的模型称为model；
# （2）令Uf=Lable-model，model的输出是label的预测值label_pre1，Label指的是数据集中的最后一列label,此处用每一条记录计算得到Uf，并对Uf进行保存。
# （4）对causal.csv数据集中的第n列中的数值删除，并用生成的随机数填充，随机数不为负数，其他列数值不变，形成不含label的新数据集new_causal.csv。
# （5）将新数据集new_causal.csv中的每一行输入model中，得到label_pre2，令label_pre3=label_pre2+Uf，输出label_pre3的值，label_pre3的值要四舍五入保存成整数
# （6）将新数据集new_causal.csv和label_pre3对应合并起来，保存到AUG_dataset.csv文件中。
# （7）将AUG_dataset.csv文件的23列数据保持不变，在最前面插入2列，数值用随机数填充，将新的数据集文件共25列保存为AUG_upsample_dataset.csv

#该部分的输出：MLP的训练损失图，前10项的Uf,干预后带label的数据集，干预后不带label的数据集，升维后的数据集。
# 超参数：干预的列数

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  # 以 'TkAgg' 为例；如有需要，尝试其他后端
import matplotlib.pyplot as plt
from datetime import datetime
import os



# Step 1: Load and preprocess the data
data = pd.read_csv('D:/A项目文件夹/imitationProject/CGIL/caufea_mining/top_10_features_and_labels.csv')
print(f'原数据集 data 的行数和列数：{data.shape}')
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values.reshape(-1, 1)

# Step 2: Define the neural network
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        # Modify the architecture based on your specific requirements
        self.fc1 = nn.Linear(10, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 128)
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(128, 64)
        self.relu = nn.ReLU()
        self.fc4 = nn.Linear(64, 10)
        self.relu = nn.ReLU()
        self.fc5 = nn.Linear(10, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        x = self.relu(x)
        x = self.fc5(x)
        return x

# Step 3: Train the neural network
def train_model_with_print(model, X_train, y_train, epochs=1000, lr=0.001, print_every=10):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    losses = []

    for epoch in range(epochs):
        inputs = torch.Tensor(X_train).float()
        labels = torch.Tensor(y_train).float()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        if (epoch + 1) % print_every == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

    return losses

# Step 4: Create and train the neural network
model = NeuralNetwork()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

losses = train_model_with_print(model, X_train, y_train,print_every=10)

# Save the trained model parameters
torch.save(model.state_dict(), 'trained_model.pth')

# Plot the loss curve
plt.plot(losses)
plt.title('Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

# Step 5: Save Uf
Uf = y - model(torch.Tensor(X)).detach().numpy()
# 打印前10项Uf
print("前10项 Uf[仅作为展示使用]:")
print(Uf[:10])
print(f'Uf的行数和列数：{Uf.shape}')

# Step 6: Generate new dataset and predict label_pre3

# n=0 #选取干预的列数 这里的索引是从0开始
new_data_do = data.copy()  # 复制原始数据集以避免修改原始数据
##注意：离散的数据干预的时候只能取离散值，连续的数据干预的时候可以取连续值。


##连续
# 生成在 [a, b) 范围内的随机数
# a = 1  # 起始值
# b = 3  # 终止值
# new_data_do.iloc[:, n] = a + (b - a) * np.random.rand(len(data))

##离散数据
new_data_do.iloc[:,9]=1#选择new_data_do的第一列赋值为1。

#new_data_do.iloc[:, n] = np.random.randn(len(data))  # 用生成的随机数替代第一列，[0,1)内均匀分布的随机数
new_data_do.to_csv('new_causal_do.csv', index=False) #do干预后的新数据集；
print(f'新数据集 new_causal_do 的行数和列数：{new_data_do.shape}')#（1910,23）

new_data_do_X = data.iloc[:, :-1] #不包含label的新数据集
new_data_do_X.to_csv('new_causal_do_X.csv', index=False)

#使用已经训练好的模型对提取出的新数据进行预测，并将预测结果存储在label_pre2中。
label_pre2 = model(torch.Tensor(new_data_do_X.values)).detach().numpy()
label_pre3 = label_pre2 + Uf
label_pre3 = np.round(label_pre3).astype(int)
print(f'数据集 new_data_do_X的行数和列数：{new_data_do_X.shape}')
print(f'label_pre3的行数和列数：{label_pre3.shape}')

# Step 7: Save the augmented dataset
# 此部分是do后的数据+新的label，数据集合并
AUG_dataset = pd.concat([new_data_do_X, pd.DataFrame(label_pre3, columns=['label_pre3'])], axis=1)
AUG_dataset.to_csv('AUG_dataset.csv', index=False)
print(f'数据集 AUG_dataset 的行数和列数：{AUG_dataset.shape}')

# Step 8: Create and save AUG_upsample_dataset
#升维,需要升几个维度
D=12
# AUG_upsample_dataset = pd.concat([pd.DataFrame(np.random.rand(len(AUG_dataset), 2)), AUG_dataset], axis=1)
a = 0  # 范围的起始值
b = 3  # 范围的终止值
# [0,3)之间均匀分布的随机数
AUG_upsample_dataset = pd.concat([pd.DataFrame((b - a) * np.random.rand(len(AUG_dataset), D) + a), AUG_dataset], axis=1)
AUG_upsample_dataset.to_csv('AUG_upsample_dataset_9_1.csv', index=False)
print(f'数据集 AUG_upsample_dataset_1 的行数和列数：{AUG_upsample_dataset.shape}')

# timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
# output_csv_path = 'D:/A项目文件夹/imitationProject/data/'+ timestamp
# #os.makedirs(output_csv_path, exist_ok=True)
# # 拼接保存文件的路径
# output_file_path = os.path.join(output_csv_path, 'AUG_upsample_dataset.csv')
# AUG_upsample_dataset.to_csv(output_file_path, index=False, header=True)
# print(f'AUG_upsample_dataset saved to {output_file_path}')