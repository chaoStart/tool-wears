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
from AllModel import CNN_1D
plt.rcParams['font.sans-serif'] = ['SimHei']  ## 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  ## 用来正常显示负号

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
print(wear_ndarray[:5,0])
wear = torch.tensor(wear_ndarray,dtype=torch.float32)
wear=wear.cuda()
# feature=feature.cuda()
print(feature.shape)
print(wear.shape)
batch_size = 10
epoch = 1000
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
print('batch_size的大小:',batch_size)

# print('/////////////////下面是训练集train进行损失函数的计算////////////////')
# for i in range(epoch):
#     model.train()
#     loss_train = 0
#     for train_x, train_y in train_loader:
#         out_train = model(train_x)
#         loss = loss_func(out_train, train_y)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         scheduler.step()
#         loss_train += loss.item()
#     loss_train /= len(train_loader)
#     train_loss.append(loss_train)
#     loss_val = 0
#     print('/////////////////下面是验证集val进行损失函数的计算////////////////')
#     with torch.no_grad():
#         model.eval()
#         for val_x, val_y in val_loader:
#             out_val = model(val_x)
#             print(val_x.is_cuda, out_val.is_cuda)
#             loss_ = loss_func(out_val, val_y)
#             loss_val += loss_.item()
#         loss_val /= len(val_loader)
#     print('epoch %s train_loss:%.6f val_loss:%6f' % (i, loss_train, loss_val))
# torch.save(model, 'CNNmodel.pkl')
print('模型已经保存~~~')
print('模型已经加载~~')
model=torch.load('CNNmodel.pkl')
# 损失曲线
# plt.figure(figsize=(8, 4))#设置画出图像的大小
# plt.figure()#设置画出图像的大小
# plt.plot(train_loss,label='train_loss')
# plt.plot(loss_val,label='val_loss')
# plt.legend()
# plt.title('CNN训练集和测试集的损失函数')
# plt.savefig('CNN训练集和验证集的loss')
# plt.show()
# 预测C1_pred
print('/////////////////下面是测试集test的计算////////////////')
model.eval()
with torch.no_grad():
    # detach有什么用?
    # 如果A网络的输出被喂给B网络作为输入， 如果我们希望在梯度反传的时候只更新B中参数的值，
    # 而不更新A中的参数值，这时候就可以使用detach()
    # .cpu()将数据的处理设备从其他设备（如.cuda()拿到cpu上），不会改变变量类型，转换后仍然是Tensor变量。
    c1_pred = model(c1_feature).detach().cpu().numpy()
    c4_pred = model(c4_feature).detach().cpu().numpy()
    c6_pred = model(c6_feature).detach().cpu().numpy()
numpy.save('CNN_c1_PRED', c1_pred)
numpy.save('CNN_c4_PRED', c4_pred)
numpy.save('CNN_c6_PRED', c6_pred)
print('预测的pred已经保存好了~')
#反归一化
c1_pred=scaler.inverse_transform(c1_pred)
# c4_pred=scaler.inverse_transform(c4_pred)
# c6_pred=scaler.inverse_transform(c6_pred)
print('c1_pred反归一化的数据类型:',type(c1_pred),c1_pred.shape)
rmse1 = numpy.sqrt(mean_squared_error(c1_pred, c1_actual))
# rmse2 = numpy.sqrt(mean_squared_error(c4_pred, c4_actual))
# rmse3 = numpy.sqrt(mean_squared_error(c6_pred, c6_actual))
print('c1 rms :%s' % rmse1)
# print('c4 rms :%s' % rmse2)
# print('c6 rms :%s' % rmse3)
print('pred的第一个刀具形状和类型:',type(c1_pred),c1_pred.shape)
print(c1_pred[:5,:])
print('actual的第一个刀具形状和类型:',type(c1_actual),c1_actual.shape)
print(c1_actual[:5,:])

print('actual的第一个刀具磨损值:',c1_actual[:,0:1].shape)
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
# c4_test=c4_actual[:,0:1]
# c4_pre=c4_pred[:,0:1]
# print('下面是计算C4第一个刀具RMSE的相关系数')
# print(numpy.sqrt(mean_squared_error(c4_test,c4_pre)))
# print('下面是计算C4第一个刀具MAE的相关系数')
# print(mean_absolute_error(c4_test,c4_pre))
# print('下面是计算C4第一个刀具R的相关系数')
# print(r2_score(c4_test,c4_pre))
# # //////////////
# c6_test=c6_actual[:,0:1]
# c6_pre=c6_pred[:,0:1]
# print('下面是计算C6第一个刀具RMSE的相关系数')
# print(numpy.sqrt(mean_squared_error(c6_test,c6_pre)))
# print('下面是计算C6第一个刀具MAE的相关系数')
# print(mean_absolute_error(c6_test,c6_pre))
# print('下面是计算C6第一个刀具R的相关系数')
# print(r2_score(c6_test,c6_pre))

x = numpy.arange(315)
plt.figure()#设置画出图像的大小
# plt.subplot(1, 3, 1)
# plt.title('No.1 tool wear')
plt.title('CNN model')
plt.plot(x, c1_actual[:,0:1],label='actual')
plt.plot(x, c1_pred[:,0:1],label='predict')
plt.xlabel('Times of cutting')
plt.ylabel(r'Average wear$\mu m$')
# 手动设置标签位置为左上角
plt.legend(loc='upper left')
plt.savefig("CNN_pred.svg" )
plt.show()

# # plt.subplot(1, 3, 2)
# plt.title('No.1 tool wear')
# plt.plot(x, c4_actual[:,0:1],label='actual')
# plt.plot(x, c4_pred[:,0:1],label='pred')
# plt.xlabel('Times of cutting')
# plt.ylabel(r'Average wear$\mu m$')
# # 手动设置标签位置为左上角
# plt.legend(loc='upper left')
# plt.show()
#
# # plt.subplot(1, 3, 3)
# plt.title('No.1 tool wear')
# plt.plot(x, c6_actual[:,0:1],label='actual')
# plt.plot(x, c6_pred[:,0:1],label='predict')
# plt.xlabel('Times of cutting')
# plt.ylabel(r'Average wear$\mu m$')
# # 手动设置标签位置为左上角
# plt.legend(loc='upper left')
# plt.show()