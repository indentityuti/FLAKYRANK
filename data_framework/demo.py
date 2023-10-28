
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix
import torch
from torch.autograd import Variable
import torch.utils.data as Data
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import torchvision
from imblearn.over_sampling import SMOTE
from imblearn.ensemble import EasyEnsembleClassifier
from imblearn.over_sampling import ADASYN,BorderlineSMOTE,SVMSMOTE
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import torch.nn as nn
X,y = make_classification(n_samples=20000,
                          n_features=40,
                          n_informative=20,
                          n_redundant=20,
                          weights=[0.98,0.02],
                          random_state=42)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42,stratify=y)
print(y.sum(),y.__len__())
process = pd.DataFrame(X_train,columns=[f'fea{i}' for i in range(1,X_train.shape[1] + 1)])
process['target'] = y_train

X_for_generate = process.query("target == 1").iloc[:,:-1].values
X_non_default = process.query('target == 0').iloc[:,:-1].values
X_for_generate = torch.tensor(X_for_generate).type(torch.FloatTensor)

n_generate = X_non_default.shape[0] - X_for_generate.shape[0]
# 超参数
BATCH_SIZE = 50
LR_G = 0.0001  # G生成器的学习率
LR_D = 0.0001  # D判别器的学习率
N_IDEAS = 20  # G生成器的初始想法(随机灵感)

# 搭建G生成器
G = nn.Sequential(  # 生成器
    nn.Linear(N_IDEAS, 128),  # 生成器等的随机想法
    nn.ReLU(),
    nn.Linear(128, 40),
)

# 搭建D判别器
D = nn.Sequential(  # 判别器
    nn.Linear(40, 128),
    nn.ReLU(),
    nn.Linear(128, 1),
    nn.Sigmoid(),  # 转换为0-1
)

# 定义判别器和生成器的优化器
opt_D = torch.optim.Adam(D.parameters(), lr=LR_D)
opt_G = torch.optim.Adam(G.parameters(), lr=LR_G)

# GAN训练
for step in range(100000):
    # 随机选取BATCH个真实的标签为1的样本
    chosen_data = np.random.choice((X_for_generate.shape[0]), size=(BATCH_SIZE), replace=False)
    artist_paintings = X_for_generate[chosen_data, :]
    # 使用生成器生成虚假样本
    G_ideas = torch.randn(BATCH_SIZE, N_IDEAS, requires_grad=True)
    G_paintings = G(G_ideas)
    # 使用判别器得到判断的概率
    prob_artist1 = D(G_paintings)
    # 生成器损失
    G_loss = torch.mean(torch.log(1. - prob_artist1))
    opt_G.zero_grad()
    G_loss.backward()
    opt_G.step()

    prob_artist0 = D(artist_paintings)
    prob_artist1 = D(G_paintings.detach())
    # 判别器的损失
    D_loss = - torch.mean(torch.log(prob_artist0) + torch.log(1. - prob_artist1))
    opt_D.zero_grad()
    D_loss.backward(retain_graph=True)
    opt_D.step()

fake_data = G(torch.randn(n_generate,N_IDEAS)).detach().numpy()
X_default = pd.DataFrame(np.concatenate([X_for_generate,fake_data]),columns=[f'fea{i}' for i in range(1,X_train.shape[1] + 1)])
X_default['target'] = 1
X_non_default = pd.DataFrame(X_non_default,columns=[f'fea{i}' for i in range(1,X_train.shape[1] + 1)])
X_non_default['target'] = 0
train_data_gan = pd.concat([X_default,X_non_default])

X_gan = train_data_gan.iloc[:,:-1]
y_gan = train_data_gan.iloc[:,-1]

print(X_gan.shape,y_gan.shape)
# output
# (27312, 40) (27312,)
