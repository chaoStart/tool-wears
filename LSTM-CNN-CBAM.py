import numpy as np
import pandas as pd
import torch
from keras import Input
from keras.layers import BatchNormalization, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from torch.utils.data import TensorDataset, random_split, DataLoader
from Model import CNN_1D
from Model import CBAM
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = 'SimHei'
//据说GitHub没法传大于50MB的数据
batch_size=10
epoch =5
lr = 0.001
# 这些输入data_x1是归一化处理过的
data_x1 = np.load("G:\pycharm\chap06_刀具磨损预测\PHM_2010\c1(31575000).npy")
data_x4 = np.load("G:\pycharm\chap06_刀具磨损预测\PHM_2010\c4(31575000).npy")
data_x6 = np.load("G:\pycharm\chap06_刀具磨损预测\PHM_2010\c6(31575000).npy")
data_y1 = np.load("G:\pycharm\chap06_刀具磨损预测\PHM_process\data_y1.npy")
data_y4 = np.load("G:\pycharm\chap06_刀具磨损预测\PHM_process\data_y4.npy")
data_y6 = np.load("G:\pycharm\chap06_刀具磨损预测\PHM_process\data_y6.npy")
print('data_x1的形状和大小:',data_x1.shape)
print('data_y1的形状:',data_y1.shape)#data_y1是没有归一化处理的数据
train_x = np.append(data_x1,data_x4, axis=0)
train_x = np.append(train_x,data_x6, axis=0)
train_y = np.append(data_y1,data_y4, axis=0)
train_y = np.append(train_y,data_y6, axis=0)
print('train_x:',train_x.shape)
# 归一化
scaler = MinMaxScaler()
train_y=scaler.fit_transform(train_y)
print('train_y的形状:',train_y.shape)
# print('归一化后train_y的数据\n',train_y[:5,:])

# 下面对数据转化为tensor的数据类型
train_x = torch.tensor(train_x,dtype=torch.float32)
train_x=train_x.cuda()
train_y = torch.tensor(train_y,dtype=torch.float32)
train_y=train_y.cuda()
print('train_x的形状:',type(train_x),train_x.shape)
print('train_y的形状:',type(train_y),train_y.shape)

# 这里是把输入x和结果y合并在一起，构成一个新的数组
dataset = TensorDataset(train_x, train_y)
print('dataset的数据形式:',type(dataset))
print('dataset的数据形式:',len(dataset))
train_num = int(len(train_y) * 0.8)
print('train_num的数据量:',len(train_y),train_num)
train_set, val_set = random_split(dataset, [train_num, len(train_y) - train_num])
print(type(train_set))
print('train_loader的形状',len(train_set))
print('val_loader的形状',len(val_set))

train_loader = DataLoader(train_set, batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size, shuffle=True)
print('train_loader的形状',type(train_loader),len(train_loader))
print('val_loader的形状',type(val_loader),len(val_loader))
# model=CNN_1D()
# for batch_data in train_loader:
#     # batch_data 是一个批次的数据，通常是一个包含输入张量和标签张量的元组
#     inputs, labels = batch_data
#     print('inputs.shape:',inputs.shape,inputs.size(0))
model=CBAM()
model = model.cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
scheduler=torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.9)
loss_func = nn.MSELoss()
train_loss = []
val_loss = []
for i in range(epoch):
    loss_train = 0
    for train_x, train_y in train_loader:
        print('-------------------------------')
        print("train_y和train_x的",type(train_y),type(train_x))
        print("train_y和train_x的形状",train_y.shape,train_x.shape)
        model.train()
        optimizer.zero_grad()
        #前向传播
        out_train = model(train_x)
        print("out_train和train_y的形状",out_train.shape,train_y.shape)
        print('-------------------------------')
        #计算损失
        loss = loss_func(out_train, train_y)
        #反向传播及优化
        loss.backward()
        optimizer.step()
        #调整学习率
        scheduler.step()
        #累计总的传递误差
        loss_train += loss.item()
    #计算均误差
    loss_train /= len(train_loader)
    train_loss.append(loss_train)

    loss_val = 0
    with torch.no_grad():
        model.eval()
        for val_x, val_y in val_loader:
            #在验证集上面进行验证测试
            out_val = model(val_x)
            loss_ = loss_func(out_val, val_y)
            # print('loss_是每次测量的损失值loss',type(loss_),loss_)
            loss_val += loss_.item()
        loss_val /= len(val_loader)
        val_loss.append(loss_val)
    print('epoch %s train_loss:%.6f val_loss:%6f' % (i, loss_train, loss_val))
# 保存模型
# torch.save(model.state_dict(), 'CBAM_model.pth')
print('模型保存成功~~~')
# 损失曲线
plt.figure(figsize=(8, 4))
plt.plot(train_loss,label='train_loss')
plt.plot(val_loss,label='val_loss')
plt.legend()
plt.title('训练集和测试集的损失函数')
plt.show()
plt.savefig('G:\pycharm\chap04_大模型学习')
