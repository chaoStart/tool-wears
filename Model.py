# 卞庆朝
# 开发时间：2024/4/17 11:59
import torch
import torch.nn as nn
# 一维卷积模块
class CNN_1D(nn.Module):
    def __init__(self):
        super(CNN_1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=7, out_channels=32, kernel_size=2, stride=1,padding=1)
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=1,padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=1,padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=4, stride=1,padding=1)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=7, kernel_size=3, stride=1,padding=1)
        self.pool3 = nn.MaxPool1d(kernel_size=3, stride=1,padding=1)
        # self.conv4 = nn.Conv1d(in_channels=32, out_channels=16, kernel_size=2, stride=1)
        # self.pool4 = nn.MaxPool1d(kernel_size=2, stride=2)
        # self.conv5 = nn.Conv1d(in_channels=16, out_channels=16, kernel_size=2, stride=1)
        # self.pool5 = nn.MaxPool1d(kernel_size=3, stride=3)
        # self.conv6 = nn.Conv1d(in_channels=16, out_channels=16, kernel_size=2, stride=1)
        # self.pool6 = nn.MaxPool1d(kernel_size=3, stride=3)
        self.net = nn.Sequential(self.conv1,nn.ReLU(),self.pool1,
                                 self.conv2,nn.ReLU(),self.pool2,
                                 self.conv3,nn.ReLU(),self.pool3)
        # self.fc = nn.Sequential(nn.Linear(in_features=224, out_features=1), nn.ReLU())
    def forward(self, x):
        print('Conv1D输入的x是:',x.shape)
        x = self.net(x)
        print('CNN_1D输出x的形状:',x.shape)
        # print('x.size(0)的形状:',type(x.size(0)),x.size(0))
        # print('x.view(x.size(0), -1)的形状:',x.view(x.size(0), -1).shape)
        # view()的作用相当于numpy中的reshape，重新定义矩阵的形状
        # out = self.fc(x.view(x.size(0), -1))#view中一个参数定为-1，代表动态调整这个维度上的元素个数，以保证元素的总数不变
        # print('out的形状:',out.shape)
        return x

# 通道注意力模块
class ChannelAttentionModule(nn.Module):
    # def __init__(self, channel=256, reduction=16):
    # def __init__(self, channel=10, reduction=2):
    def __init__(self, channel=10, reduction=2):
        print('传参成功了吗？',channel)
        super(ChannelAttentionModule, self).__init__()
        mid_channel = channel // reduction #mid_channel==5
        self.avg_pool = nn.AdaptiveAvgPool2d(1)#平均池化，使用自适应池化缩减map的大小，保持通道不变
        self.max_pool = nn.AdaptiveMaxPool2d(1)#最大池化，
        # 两个卷积层用于从池化后的特征中学习注意力权重
        self.shared_MLP = nn.Sequential(
            nn.Linear(in_features=channel, out_features=mid_channel),
            nn.ReLU(),
            nn.Linear(in_features=mid_channel, out_features=channel)
        )
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        print('channel的输入:',x.shape)
        avgout = self.shared_MLP(self.avg_pool(x).view(x.size(0), -1)).unsqueeze(2).unsqueeze(3)#对平均池化的特征进行处理
        print('avg',avgout.size())
        maxout = self.shared_MLP(self.max_pool(x).view(x.size(0), -1)).unsqueeze(2).unsqueeze(3)#对最大池化的特征进行处理
        print('max',maxout.size())
        return self.sigmoid(avgout + maxout) # 将两种池化的特征加权和作为输出

# 空间注意力模块
class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        print('**********************')
        print('spatial的输入:', x.shape)
        avgout = torch.mean(x, dim=1, keepdim=True)    # 平均池化，map尺寸不变，缩减通道
        print('savg',avgout.size())  # [16, 1, 384, 384]
        maxout, _ = torch.max(x, dim=1, keepdim=True)  # 最大池化
        print('smax',maxout.size())  # [16, 1, 384, 384]
        out = torch.cat([avgout, maxout], dim=1)       # 将两种池化的特征图连接起来
        print('平均+Max:',out.size())  # [16, 2, 384, 384]
        out = self.sigmoid(self.conv2d(out)) # 通过卷积层处理连接后的特征图+++使用sigmoid激活函数计算注意力权重
        print('ssigmoid',out.size())
        return out

# CBAM模块
class CBAM(nn.Module):
    # def __init__(self, in_channel=256, out_channels=64):
    def __init__(self,in_channel=7, out_channels=7):
        super(CBAM, self).__init__()
        # self.channel=channel
        self.cnn_1d=CNN_1D()
        self.channel_attention = ChannelAttentionModule()
        self.spatial_attention = SpatialAttentionModule()
        self.relu = nn.ReLU(inplace=True)
        # self.conv = nn.Conv2d(in_channel, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv1 = nn.Conv1d(in_channels=7, out_channels=32, kernel_size=7, stride=3)
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=2)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=out_channels, kernel_size=3, stride=2)
        self.pool2 = nn.MaxPool1d(kernel_size=3, stride=2)
        self.bn = nn.BatchNorm1d(num_features=out_channels)
        self.fc = nn.Sequential(nn.Linear(in_features=1449, out_features=1), nn.ReLU())

    def forward(self, x):
        print('CBAM输入的x是:', x.shape)
        out=self.cnn_1d(x)
        out=out.unsqueeze(0)# 增加一个维度
        print('cnn1D输出的out是:', out.shape)
        print('out[1]是多少:',out.size(1))
        out1 = self.channel_attention(out)
        print('channel_attention输出的out1是:', out1.shape)
        out = out1 * out
        print('out1*out乘积的输出:',out.shape)
        # out = self.channel_attention(out) * out
        out = self.spatial_attention(out) * out
        residual = x
        residual = residual.unsqueeze(0)  # 增加一个维度
        print('Residual增加一个维度的Residual是:', residual.shape)
        out = residual + out
        print('res+channel+spatial=out的输出:',out.size())
        out=torch.squeeze(out, dim=0)
        print('新的维度:',out.shape)
        out = self.relu(out)
        out = self.conv1(out)
        out=self.pool1(out)
        out = self.conv2(out)
        out=self.pool2(out)
        print('conv12(out)+pool12(out)输出后的维度:',out.shape)
        print('out.view(x.size(0), -1)输出后的维度:',out.view(x.size(0), -1).shape)
        out = self.fc(out.view(x.size(0), -1))#view中一个参数定为-1，代表动态调整这个维度上的元素个数，以保证元素的总数不变
        # out = self.bn(out)
        return out