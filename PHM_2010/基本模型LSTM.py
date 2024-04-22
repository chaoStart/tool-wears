# 卞庆朝
# 开发时间：2023/3/2 9:52
import torch
import numpy as np
import torch.nn as nn
import torch.utils.data as Data
import math
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from AllModel import LSTM
plt.rcParams['font.sans-serif'] = 'SimHei'
EPOCH = 500
BATCH_SIZE = 128
LR = 0.002
data_x1 = np.load("G:\pycharm\chap06_刀具磨损预测\PHM_process\data_x1.npy")
data_x4 = np.load("G:\pycharm\chap06_刀具磨损预测\PHM_process\data_x4.npy")
data_x6 = np.load("G:\pycharm\chap06_刀具磨损预测\PHM_process\data_x6.npy")
data_y1 = np.load("G:\pycharm\chap06_刀具磨损预测\PHM_process\data_y1.npy")
data_y4 = np.load("G:\pycharm\chap06_刀具磨损预测\PHM_process\data_y4.npy")
data_y6 = np.load("G:\pycharm\chap06_刀具磨损预测\PHM_process\data_y6.npy")
print('data_x1的形状:',data_x1.shape)
print('data_y1的形状:',data_y1.shape)
print('data_y1的数据:\n',data_y1[:10])
scaler = MinMaxScaler()
def normalization(data):#数据标准化
    # print('特征向量数据data输入时的形状:',type(data))
    mu = np.mean(data, axis=0)#每列取平均值
    sigma = np.std(data, axis=0)#每列取标准差
    return (data - mu) / sigma#
def norm_all(data):#对所有数据进行标准化
    d = np.empty((data.shape[0], data.shape[1], data.shape[2]))
    # print('data.shape[0]=315',data.shape[0])
    # print('data.shape[1]=6',data.shape[1])
    # print('data.shape[2]=24',data.shape[2])
    for i in range(data.shape[1]):
        data1 = data[:, i, :]#取出来的data1是6*(315,24)
        print('**************第外层{}次***************'.format(i), data1.shape)
        for j in range(data1.shape[0]):
            data2 = data1[j, :]
            d[j, i, :] = normalization(data2)
    return d
def normal_label(data):
    # 对y的数据进行标准化
    print('标准化的数据类型:',data.shape)
    # A.min(0) : 返回A每一列最小值组成的一维数组；
    data=data.squeeze(1)
    print('squeeze后的数据类型:', data.shape)
    minVals = data.min(0)
    maxVals = data.max(0)
    print('minVals和maxVals大小:',minVals,maxVals)
    ranges = maxVals - minVals
    # normData = np.zeros(np.shape(data))
    m = data.shape[0]
    normData = data - np.tile(minVals, (m, 1))
    normData = normData / np.tile(ranges, (m, 1))
    print('normData的数据类型:',normData.shape[0])
    # 归一化
    # scaled = scaler.fit_transform(data)
    # print('scaled的形状:',scaled.shape)
    return normData[0]
# 获取第一次预处理得到的24个特征向量
# 对输入数据进行预处理
print('读取data_x1的数据,进入第一次循环')
data_x1 = norm_all(data_x1)
print('读取data_x4的数据,进入第二次循环')
data_x4 = norm_all(data_x4)
print('读取data_x6的数据,进入第三次循环')
data_x6 = norm_all(data_x6)
print('data_x1的数据类型:',type(data_x1),data_x1.shape)
print('data_x4的数据类型:',type(data_x4),data_x4.shape)
print('data_x6的数据类型:',type(data_x6),data_x6.shape)
# 对预测值y做了标准化处理
data_y1 = normal_label(data_y1)
data_y4 = normal_label(data_y4)
data_y6 = normal_label(data_y6)
print('data_y1的数据类型:',type(data_y1),data_y1.shape)
print('data_y4的数据类型:',type(data_y4),data_y4.shape)
print('data_y6的数据类型:',type(data_y6),data_y6.shape)
# np.append[axis=0]表示两个数组上下合并
train_x = np.append(data_x4, data_x6, axis=0)
train_y = np.append(data_y4, data_y6, axis=0)
test_x = data_x1
test_y = data_y1
print('train_x的数据类型:',type(train_x),train_x.shape)
print('train_y的数据类型:',type(train_y),train_y.shape)
print('test_x的数据类型:',type(test_x),test_x.shape)
print('test_y的数据类型:',type(test_y),test_y.shape)
# torch.from_numpy()用来将数组array转换为张量Tensor
train_x,train_y= torch.from_numpy(train_x),torch.from_numpy(train_y)
# train_y = torch.from_numpy(train_y)
test_x,test_y= torch.from_numpy(test_x),torch.from_numpy(test_y)
# test_y = torch.from_numpy(test_y)
print('torch.from_numpy将数组array转化为tensor的数据类型:',type(train_x),train_x.shape)
print('torch.from_numpy将数组array转化为tensor的数据类型:',type(train_y),train_y.shape)
# 对给定的 tensor 数据，将他们包装成 dataset
train_dataset = Data.TensorDataset(train_x, train_y)
print('train_dataset是什么:',type(train_dataset),len(train_dataset))
all_num = train_x.shape[0]
print('all_num的数据类型:',train_x.shape[0])
train_num = int(all_num * 0.8)
print('train_num的数据形状',train_num)
train_data, val_data = Data.random_split(train_dataset, [train_num, all_num - train_num])
print('train_data的数据类型和形状',type(train_data),len(train_data))
print('val_data的数据类型和形状',type(val_data),len(val_data))
# torch.utils.data.DataLoader(): 构建可迭代的数据装载器, 我们在训练的时候，每一个for循环，
# 每一次iteration，就是从DataLoader中获取一个batch_size大小的数据的。
train_loader = Data.DataLoader(
    # 把训练数据集504个放在train_loader
    dataset=train_data, batch_size=BATCH_SIZE, shuffle=True, )
val_loader = Data.DataLoader(
    # 把测试数据集126个放在val_loader
    dataset=val_data, batch_size=BATCH_SIZE, shuffle=True, )
test_dataset = Data.TensorDataset(test_x, test_y)
test_loader = Data.DataLoader(
    dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False, )
print('test_dataset的数据类型和形状',type(test_dataset),len(test_dataset))
model = LSTM()
# 使用cuda的GPU来加速
if torch.cuda.is_available():
    model = model.cuda()
# 优化器的优化参数类型,以及设置学习率
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.9)
loss_func = torch.nn.MSELoss()
train_loss = []
val_loss = []
lr_list = []
# EPOCH = 500
BATCH_SIZE =128
LR = 0.002
for epoch in range(EPOCH):
    total_loss = 0
    total_loss2 = 0
    # model.train()的作用是启用 Batch Normalization 和 Dropout。
    model.train()
    # 从504个train_loader里面抽取x,y
    print('////////////下面是计算训练集train的损失值////////////////')
    for step, (b_x, b_y) in enumerate(train_loader):
        b_x = b_x.float()
        b_y = b_y.float()
        if torch.cuda.is_available():
            b_x = b_x.cuda()
            b_y = b_y.cuda()
        # torch.squeeze(input, dim=None, out=None)：对数据的维度进行压缩，去掉维数为1的的维度。
        output = model(b_x).squeeze(-1)
        loss = loss_func(output, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        #  .cpu()将数据的处理设备从其他设备（如.cuda()拿到cpu上），不会改变变量类型，转换后仍然是Tensor变量。
        #  .item()将一个Tensor变量转换为python标量（int float等）常用于用于深度学习训练时，
        # 将loss值转换为标量并加，以及进行分类任务，计算准确值值时需要
        total_loss += loss.cpu().item()
    total_loss /= len(train_loader.dataset)
    train_loss.append(total_loss)
    print('关于train_loss的长度',len(train_loss))
    # 上面是计算训练集的平均损失值
    lr_list.append(optimizer.state_dict()['param_groups'][0]['lr'])
    # model.eval()的作用是不启用 Batch Normalization 和 Dropout。
    model.eval()
    print('////////////下面是计算验证集val的损失值////////////////')
    with torch.no_grad():
        for i, (v_x, v_y) in enumerate(val_loader):
            v_x = v_x.float()
            v_y = v_y.float()
            if torch.cuda.is_available():
                v_x = v_x.cuda()
                v_y = v_y.cuda()
            test_output = model(v_x).squeeze(-1)
            v_loss = loss_func(test_output, v_y)
            total_loss2 += v_loss.cpu().item()
    total_loss2 /= len(val_loader.dataset)
    val_loss.append(total_loss2)
    print('Train Epoch: {} \t Train Loss:{:.6f} \t Val Loss:{:.6f}'.format(epoch, total_loss, total_loss2))
# torch.save(model, 'LSTMmodel.pkl')
print('模型已经保存~~~')
X0 = np.array(train_loss).shape[0]

print('X0是什么:',X0)
print('train_loss的形状和大小:',len(train_loss),train_loss)
print('val_loss的形状和大小:',len(val_loss),val_loss)
# x1 = range(0, X0)
# x2 = range(0, X0)
# y1 = train_loss
# y2 = val_loss
# plt.subplot(2, 1, 1)
# plt.plot(x1, y1, '-')
# plt.ylabel('train_loss')
# plt.subplot(2, 1, 2)
# plt.plot(x2, y2, '-')
# plt.ylabel('val_loss')
# plt.show()

# /////////////////下面是计算测试集的损失函数////////////////////
print('/////////////////下面是计算测试集的损失函数////////////////////')
pred = torch.empty(1)#创建1个空数组tensor
model.eval()
# model.eval()的作用是不启用 Batch Normalization 和 Dropout。
with torch.no_grad():
# 这里的test_loader其实是C1工况的数据集，前序内容train_loader、val_loader是将C4、C6数据集作为训练集和验证集
    for i, (tx, ty) in enumerate(test_loader):
        tx = tx.float()
        ty = ty.float()
        if torch.cuda.is_available():
            tx = tx.cuda()
            ty = ty.cuda()
        out = model(tx).squeeze(-1)
        # .item() 将一个Tensor变量转换为python标量（int float等）常用于用于深度学习训练时，
        # 将loss值转换为标量并加，以及进行分类任务，计算准确值值时需要
        pred = torch.cat((pred, out.cpu()))
        # detach有什么用?
        # 如果A网络的输出被喂给B网络作为输入， 如果我们希望在梯度反传的时候只更新B中参数的值，
        # 而不更新A中的参数值，这时候就可以使用detach()
        # numpy.delete(arr, obj, axis=None)
        # arr: 输入向量
        # obj: 表明哪一个子向量应该被移除。可以为整数或一个int型的向量
        # axis: 表明删除哪个轴的子向量，若默认，则返回一个被拉平的向量,0表示按照行删除
# y_pred表示预测的值
y_pred = np.delete(pred.detach().numpy(), 0, axis=0)#去掉了第316个数字中第一个为0的数值
print('y_pred是什么样子的',type(y_pred),y_pred.shape)
# print('去掉一个数字之后的pred',pred)
# np.save('Y_PRED', y_pred)
# y_actual表示实际的y值
y_actual = test_y.cpu().detach().numpy()
print('y_actual的数据:',y_actual.shape,y_actual)
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = 'SimHei'
xx1 = range(0, 315)
# plt.plot(xx1, y_pred, color='black', label='Predicted value')
# plt.plot(xx1, y_actual, color='red', label='Actual value')
plt.plot(xx1, y_pred,label='predict')
plt.plot(xx1, y_actual,label='actual')
plt.xlabel('Times of cutting')
plt.ylabel(r'Average wear$\mu m$')
# 手动设置标签位置为左上角
plt.legend(loc='upper left')
plt.show()

rmse = math.sqrt(mean_squared_error(y_pred, y_actual))
print('Test RMSE: %.3f' % rmse)