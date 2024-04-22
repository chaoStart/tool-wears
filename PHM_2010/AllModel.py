# 卞庆朝
# 开发时间：2023/9/8 21:19
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
# 定义LSTM模型
class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=7,
            hidden_size=64,
            num_layers=2,
            bias=True,
            batch_first=True,
            dropout=0.05,
        )
        self.out = nn.Sequential(
            nn.Linear(64, 10),
            nn.BatchNorm1d(10, momentum=0.5),
            nn.ReLU(),
            nn.Linear(10, 1),
        )

    def forward(self, x):
        # permute()函数其实是对矩阵的块行列进行交换
        # 如果想要断开这两个变量之间的依赖（x本身是contiguous的），就要使用contiguous()针对x进行变化，感觉上就是我们认为的深拷贝。
        x = x.permute(0, 2, 1).contiguous()
        r_out, (h_n, h_c) = self.lstm(x, None)
        out = self.out(r_out[:, -1, :])
        return out
# 定义CNN_1D模型
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
# 定义残差块
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out
# 定义ResNet模型
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self.make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self.make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self.make_layer(block, 512, num_blocks[3], stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

