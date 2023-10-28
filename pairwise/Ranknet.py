import os
import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from data import process_data  # 导入数据处理函数


class RankNet(nn.Module):
    def __init__(self, num_features):
        super(RankNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.output = nn.Sigmoid()

    def forward(self, input1, input2):
        s1 = self.model(input1)
        s2 = self.model(input2)
        diff = s1 - s2
        prob = self.output(diff)
        return prob

    def predict(self, input_):
        x=self.model(input_)
        x=self.output(x)
        return x

# 循环执行代码十次

start_time = time.time()
# 调用函数并获取返回的NumPy数组
data_more,data_less,data_zone_one,data_zone_two = process_data()
test_data=np.loadtxt('./dataflagg/result.txt', delimiter=',', dtype=float)
data_more = np.array(data_more)
data_less = np.array(data_less)
data_zone_one = np.array(data_zone_one)
data_zone_two = np.array(data_zone_two)
test_data = np.array(test_data)

test_data=test_data[:,1:]
data_more = torch.from_numpy(data_more).float().cuda()
data_less = torch.from_numpy(data_less).float().cuda()
data_zone_one = torch.from_numpy(data_zone_one).float().cuda()
data_zone_two = torch.from_numpy(data_zone_two).float().cuda()
test_data = torch.from_numpy(test_data).float().cuda()

print(f"the :------------data_more shape:{data_more.shape}")
print(f"the :------------data_less shape:{data_less.shape}")
print(f"the :------------data_zone_one shape:{data_zone_one.shape}")
print(f"the :------------data_zone_two shape:{data_zone_two.shape}")
print(f"the :------------test_data shape:{test_data.shape}")

# 使用gpu进行训练
more_label_size, _ = data_more.shape
print(f"label_size:{more_label_size}")
zone_label_size , _=data_zone_one.shape
print(f"data_zone:{zone_label_size}")
NUM_FEATURES = 23
# NUM_EPOCHS = 200
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 定义你的RankNet模型和优化器
ranknet = RankNet(num_features=NUM_FEATURES)
ranknet = ranknet.to(device)

optimizer = torch.optim.Adam(ranknet.parameters(), lr=0.0001)

# 定义损失函数为二元交叉熵
criterion = nn.BCELoss()
criterion = criterion.to(device)

data_labels = np.ones(more_label_size)
fu_data_labels= -np.ones(more_label_size)
zero_lable=np.zeros(zone_label_size)
one_data_labels = torch.from_numpy(data_labels).float().view(-1, 1).to(device)
zone_data_labels = torch.from_numpy(zero_lable).float().view(-1, 1).to(device)
fu_data_labels = torch.from_numpy(fu_data_labels).float().view(-1, 1).to(device)
batch_size = 16384

train_dataset_onezheng = TensorDataset(data_more, data_less, one_data_labels)
train_dataset_onefu = TensorDataset( data_less,data_more, fu_data_labels)
train_dataset_zone = TensorDataset( data_zone_one,data_zone_two, zone_data_labels)
train_loader_zheng = DataLoader(train_dataset_onezheng, batch_size=batch_size)
train_loader_fu = DataLoader(train_dataset_onefu, batch_size=batch_size)
train_loader_zero = DataLoader(train_dataset_zone, batch_size=batch_size)

NUM_EPOCHS = 400  # 假设你希望训练10个epoch

# 数据加载器
train_loaders = [train_loader_zheng, train_loader_fu, train_loader_zero]
# train_loaders = [train_loader_zheng]

# 用于存储各数据集的loss
losses = []

for loader in train_loaders:
    print(f"loader:{loader}")

    print("Start training for a dataset")
    with tqdm(total=NUM_EPOCHS, desc="Training") as pbar:
        for epoch in range(NUM_EPOCHS):
            running_loss = 0.0
            for data_more, data_less, labels in loader:
                outputs = ranknet(data_more, data_less)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()

                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            losses.append(running_loss / len(loader))

            if (epoch + 1) % 1 == 0:
                pbar.set_postfix({'Loss': running_loss / len(loader)})
                pbar.update(1)

# 训练完所有数据集后，你可以进行预测
print("Start predicting")
with torch.no_grad():
    test_outputs = ranknet.predict(test_data)
# 将预测结果保存到文件
result_filename = f'predict/result.txt'
os.makedirs(os.path.dirname(result_filename), exist_ok=True)  # 创建目录
with open(result_filename, 'w') as file:
    for prediction in test_outputs:
        # 将每个预测结果写入文件，假设 prediction 是一个标量值
        file.write(f'{prediction}\n')
# 保存预测结果到指定文件

# 保存预测结果到指定文件
# result_filename = f'predict/data_result.txt'
# os.makedirs(os.path.dirname(result_filename), exist_ok=True)  # 创建目录
# with open(result_filename, 'w') as file:
#     for j in range(len(test_outputs)):
#         result_line = f"{test_outputs[j][0]}\n"
#         file.write(result_line)
#
# # 保存 test_data_labels 到指定文件
#
# label_filename = f'./label/test_label_{i}.txt'  # 添加目录路径./label/
# os.makedirs(os.path.dirname(label_filename), exist_ok=True)  # 创建目录
# with open(label_filename, 'w') as file:
#     for label in test_data_labels:
#         file.write(f"{label}\n")

# 保存模型权重

end_time = time.time()  # 记录结束时间
elapsed_time = end_time - start_time
print(f"Time elapsed for iteration : {elapsed_time:.2f} seconds")
torch.save(ranknet.state_dict(), f'ranknet.pth')

