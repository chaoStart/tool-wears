# 卞庆朝
# 开发时间：2023/9/10 14:38
import torch
import numpy
import pandas
from torch import nn
import tensorflow as tf
from matplotlib import pyplot as plt
from torch.utils.data import random_split, TensorDataset, DataLoader
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
plt.rcParams['font.sans-serif'] = ['SimHei']  ## 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  ## 用来正常显示负号
class CNN_1D(nn.Module):
    def __init__(self):
        super(CNN_1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=7, out_channels=32, kernel_size=1, stride=1)
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=3)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=2, stride=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=2, stride=1)
        self.pool3 = nn.MaxPool1d(kernel_size=3, stride=3)
        self.conv4 = nn.Conv1d(in_channels=32, out_channels=16, kernel_size=2, stride=1)
        self.pool4 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv5 = nn.Conv1d(in_channels=16, out_channels=16, kernel_size=2, stride=1)
        self.pool5 = nn.MaxPool1d(kernel_size=3, stride=3)
        self.conv6 = nn.Conv1d(in_channels=16, out_channels=16, kernel_size=2, stride=1)
        self.pool6 = nn.MaxPool1d(kernel_size=3, stride=3)
        self.fc = nn.Sequential(nn.Linear(in_features=224, out_features=3), nn.ReLU())
        # self.net=nn.Sequential(self.conv1,nn.ReLU(),self.pool1,
        #                        self.conv5,nn.ReLU(),self.pool5,
        #                        self.conv6,nn.ReLU(),self.pool6)
        self.net = nn.Sequential(self.conv1, nn.ReLU(), self.pool1, self.conv2, nn.ReLU(), self.pool2, self.conv3,
                                 nn.ReLU(), self.pool3,
                                 self.conv4, nn.ReLU(), self.pool4, self.conv5, nn.ReLU(), self.pool5, self.conv6,
                                 nn.ReLU(), self.pool6)
    def forward(self, x):
        x = self.net(x)
        # view()的作用相当于numpy中的reshape，重新定义矩阵的形状
        out = self.fc(x.view(x.size(0), -1))#view中一个参数定为-1，代表动态调整这个维度上的元素个数，以保证元素的总数不变
        return out
c1_feature = numpy.load('G:\pycharm\chap06_刀具磨损预测\PHM_2010\c1(31575000).npy')
c1_feature = torch.tensor(c1_feature, dtype=torch.float32)
c4_feature = numpy.load('G:\pycharm\chap06_刀具磨损预测\PHM_2010\c4(31575000).npy')
c4_feature = torch.tensor(c4_feature, dtype=torch.float32)
c6_feature = numpy.load('G:\pycharm\chap06_刀具磨损预测\PHM_2010\c6(31575000).npy')
c6_feature = torch.tensor(c6_feature, dtype=torch.float32)
print('c1_feature、c4_feature、c6_feature:',numpy.array(c1_feature).shape,numpy.array(c4_feature).shape,numpy.array(c6_feature).shape)
c1_feature, c4_feature, c6_feature = c1_feature.cuda(), c4_feature.cuda(), c6_feature.cuda()
feature = torch.concat([c1_feature, c4_feature, c6_feature], dim=0)
print('feature的长度:',feature.shape)

c1_actual = pandas.read_csv('G:\pycharm\chap06_刀具磨损预测\c1_wear.csv', index_col=0, header=0,names=['flute_1','flute_2', 'flute_3'])
print('c1_actual直接padas读取数据的形状:',c1_actual.shape)
c1_actual = c1_actual.values.reshape(-1, 3)#reshape(-1,3)转化成3列，行需要计算。
print('c1_actual现在的数据形式是:',c1_actual.shape)
c4_actual = pandas.read_csv('G:\pycharm\chap06_刀具磨损预测\c4_wear.csv', index_col=0, header=0,names=['flute_1','flute_2', 'flute_3'])
c4_actual = c4_actual.values.reshape(-1, 3)#reshape(-1,3)转化成3列，行需要计算。
print('c4_actual现在的数据形式是:',c4_actual.shape)
c6_actual = pandas.read_csv('G:\pycharm\chap06_刀具磨损预测\c6_wear.csv', index_col=0, header=0,names=['flute_1','flute_2', 'flute_3'])
c6_actual = c6_actual.values.reshape(-1, 3)#reshape(-1,3)转化成3列，行需要计算。
print('c6_actual现在的数据形式是:',c6_actual.shape)
wear_ndarray=numpy.concatenate([c1_actual,c4_actual,c6_actual],axis=0)
print('全部的磨损值y的数据集:',wear_ndarray.shape)
# 下面是进行数据的归一化
scaler=MinMaxScaler()
wear_ndarray=scaler.fit_transform(wear_ndarray)
wear = torch.tensor(wear_ndarray,dtype=torch.float32)
wear=wear.cuda()
# feature=feature.cuda()
batch_size = 10
epoch = 10
lr = 0.001
# 这里是把输入x和结果y合并在一起，构成一个新的数组
dataset = TensorDataset(feature, wear)
print('dataset的数据形式:',type(dataset))
print('dataset的数据形式:',len(dataset))
train_num = int(len(wear) * 0.8)
print('wear和train_num的数据量:',len(wear),train_num)
train_set, val_set = random_split(dataset, [train_num, len(wear) - train_num])
print(type(train_set))
print('train_loader的形状',len(train_set))
print('val_loader的形状',len(val_set))
train_loader = DataLoader(train_set, batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size, shuffle=True)
print('train_loader的形状',len(train_loader))
print('val_loader的形状',len(val_loader))
model = CNN_1D()
model = model.cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
scheduler=torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.9)
loss_func = nn.MSELoss()
train_loss = []
# epoch=1500
for i in range(epoch):
    model.train()
    loss_train = 0
    for train_x, train_y in train_loader:
        out_train = model(train_x)
        loss = loss_func(out_train, train_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        loss_train += loss.item()
    loss_train /= len(train_loader)
    train_loss.append(loss_train)
    loss_val = 0
    with torch.no_grad():
        model.eval()
        for val_x, val_y in val_loader:
            out_val = model(val_x)
            print(val_x.is_cuda, out_val.is_cuda)
            loss_ = loss_func(out_val, val_y)
            loss_val += loss_.item()
        loss_val /= len(val_loader)
    # if i% 100 == 0:
    #     print('epoch %s train_loss:%.6f val_loss:%6f' % (i, loss_train, loss_val))
    print('epoch %s train_loss:%.6f val_loss:%6f' % (i, loss_train, loss_val))
# 损失曲线
plt.figure(figsize=(8, 4))
plt.plot(train_loss,label='train_loss')
plt.plot(loss_val,label='val_loss')
plt.legend()
plt.title('训练集和测试集的损失函数')
# plt.show()
plt.savefig('G:\pycharm\chap06_刀具磨损预测\PHM_2010\训练集和验证集的loss')
# 预测
with torch.no_grad():
    # detach有什么用?
    # 如果A网络的输出被喂给B网络作为输入， 如果我们希望在梯度反传的时候只更新B中参数的值，
    # 而不更新A中的参数值，这时候就可以使用detach()
    # .cpu()将数据的处理设备从其他设备（如.cuda()拿到cpu上），不会改变变量类型，转换后仍然是Tensor变量。
    c1_pred = model(c1_feature).detach().cpu().numpy()
    c4_pred = model(c4_feature).detach().cpu().numpy()
    c6_pred = model(c6_feature).detach().cpu().numpy()

c1_pred=c1_pred
c4_pred=c4_pred
c6_pred=c6_pred
c1_actual=scaler.fit_transform(c1_actual)
c4_actual=scaler.fit_transform(c4_actual)
c6_actual=scaler.fit_transform(c6_actual)
# c1_pred=scaler.inverse_transform(c1_pred)#反归一化
# c4_pred=scaler.inverse_transform(c4_pred)
# c6_pred=scaler.inverse_transform(c6_pred)
print('c1_pred反归一化的数据类型:',type(c1_pred),c1_pred.shape)
print('c1_actual反归一化的数据类型:',type(c1_actual),c1_actual.shape)
rmse1 = numpy.sqrt(mean_squared_error(c1_pred, c1_actual))
rmse2 = numpy.sqrt(mean_squared_error(c4_pred, c4_actual))
rmse3 = numpy.sqrt(mean_squared_error(c6_pred, c6_actual))

print('actual的第一个刀具形状和类型:',type(c1_actual),c1_actual.shape)
print('actual的第一个刀具磨损值:',c1_actual[:,0:1].shape)
print('pred的第一个刀具形状和类型:',type(c1_pred),c1_pred.shape)
print('pred的第一个刀具磨损值:',c1_pred[:,0:1].shape)
c1_test=c1_actual[:,0:1]
c1_pre=c1_pred[:,0:1]
print('下面是计算C1第一个刀具RMSE的相关系数')
print(numpy.sqrt(mean_squared_error(c1_test,c1_pre)))
print('下面是计算C1第一个刀具MAE的相关系数')
print(mean_absolute_error(c1_test,c1_pre))
print(numpy.mean(numpy.abs(c1_test-c1_pre)))
print('下面是计算C1第一个刀具R的相关系数')
print(r2_score(c1_test,c1_pre))
print(1-(numpy.sum((c1_test-c1_pre)**2))/numpy.sum((c1_test-numpy.mean(c1_test))**2))
# //////////////////
c4_test=c4_actual[:,0:1]
c4_pre=c4_pred[:,0:1]
print('下面是计算C4第一个刀具RMSE的相关系数')
print(numpy.sqrt(mean_squared_error(c4_test,c4_pre)))
print('下面是计算C4第一个刀具MAE的相关系数')
print(mean_absolute_error(c4_test,c4_pre))
print('下面是计算C4第一个刀具R的相关系数')
print(r2_score(c4_test,c4_pre))
# //////////////
c6_test=c6_actual[:,0:1]
c6_pre=c6_pred[:,0:1]
print('下面是计算C6第一个刀具RMSE的相关系数')
print(numpy.sqrt(mean_squared_error(c6_test,c6_pre)))
print('下面是计算C6第一个刀具MAE的相关系数')
print(mean_absolute_error(c6_test,c6_pre))
print('下面是计算C6第一个刀具R的相关系数')
print(r2_score(c6_test,c6_pre))

x = numpy.arange(315)
plt.figure(figsize=(12, 6))
# plt.subplot(1, 3, 1)
plt.plot(x, c1_actual[:,0:1],color='red',label='actual')
plt.plot(x, c1_pred[:,0:1],color='black', label='pred')
plt.legend()
# plt.title('c1')
plt.show()
# plt.subplot(1, 3, 2)
plt.plot(x, c4_actual[:,0:1],color='red',label='actual')
plt.plot(x, c4_pred[:,0:1],color='black', label='pred')
plt.legend()
# plt.title('c4')
plt.show()
# plt.subplot(1, 3, 3)
plt.plot(x, c6_actual[:,0:1],color='red', label='actual')
plt.plot(x, c6_pred[:,0:1],color='black', label='pred')
plt.legend()
# plt.title('c6')
plt.show()
plt.savefig('G:\pycharm\chap06_刀具磨损预测\PHM_2010\回归曲线')
print('c1 rms :%s' % rmse1)
print('c4 rms :%s' % rmse2)
print('c6 rms :%s' % rmse3)