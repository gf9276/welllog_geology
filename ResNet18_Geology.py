import torch
import torch.nn as nn

# training config
slicelength = 97
epoch = 100
batchsize = 1024
learningrate = 0.02
lossfunc = 0

# network config
featurenum = 6
label = ['34', '19', '3', '4', '25', '2', '1', '0', '23', '35', '32', '33', '31', '30', '28', '29', '26', '27', '22', '21', '12', '13', '14', '15', '16', '17',
         '18', '24', '11', '20', '5', '6', '7', '8', '9', '10']

# -------- Do not delete this line, the configuration ends here. --------

# --------- 变量映射 & 二次处理 & 通用函数定义, 我不喜欢原先的变量命名. ---------
model_name = 'ResNet18'
features_num = featurenum
label_classes = label
label_num = len(label_classes)
batch_size = batchsize
loss_dict = {0: torch.nn.CrossEntropyLoss(), 1: torch.nn.CrossEntropyLoss()}  # 为了兼容命令,暂时保留1
loss_func_idx = lossfunc


def loss_and_opt(net):
    loss_func = loss_dict[loss_func_idx]
    opt = torch.optim.Adam(net.parameters(), lr=learningrate)
    exp_lr = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.98)
    return loss_func, opt, exp_lr


# ---------------------------- 模型的具体内容 ----------------------------

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.stride = 1
        self.layer = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=self.stride, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),

            nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )

        if in_channels != out_channels:
            self.res_layer = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=self.stride),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.res_layer = None

    def forward(self, x):
        # x = self.layer(x)
        if self.res_layer is not None:
            residual = self.res_layer(x)
        else:
            residual = x
        return self.layer(x) + residual


class Net(nn.Module):
    def __init__(self, in_channels=features_num, classes=label_num):
        super(Net, self).__init__()
        self.model_type = "general"  # general 或者 seq2seq

        self.Layers = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            ResidualBlock(32, 32),
            ResidualBlock(32, 32),

            ResidualBlock(32, 64),
            ResidualBlock(64, 64),

            ResidualBlock(64, 128),
            ResidualBlock(128, 128),

            ResidualBlock(128, 256),
            ResidualBlock(256, 256),

            nn.AdaptiveAvgPool1d(1)
        )

        self.fc = nn.Linear(256, classes)

    def forward(self, x):
        x = self.Layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
