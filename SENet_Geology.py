"""SENet in PyTorch.

SENet is the winner of ImageNet-2017. The paper is not released yet.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# training config
slicelength = 97
epoch = 2000
learningrate = 0.002

# network config
featurenum = 5
lossfunc = 0
lossdict = {0: torch.nn.CrossEntropyLoss(), 1: torch.nn.CrossEntropyLoss()}  # 为了兼容命令,暂时保留1
label = ['36', '21', '3', '4', '27', '2', '1', '0', '25', '37', '34', '35', '33', '32', '30', '31', '28', '29', '24', '23', '13', '14', '15', '16', '17', '18',
         '19', '26', '12', '22', '5', '6', '20', '7', '8', '9', '10', '11']
trainLabel = './Data/train_地质分层_编号_LABEL.txt'
valLabel = './Data/val_地质分层_编号_LABEL.txt'
batchsize = 1024
# NAN:CONFIG-END

modelName = 'SENet'
trainDir = './Data/train.h5'
valDir = './Data/val.h5'
savePath = './Log/output.pth'

gpuId = 0
labelNum = len(label)
avgPool = 0
pretrainedPath = "./Log/output_save.pth"


def loss_and_opt(net):
    loss_func = lossdict[lossfunc]
    opt = torch.optim.Adam(net.parameters(), lr=learningrate)
    exp_lr = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.995)
    # exp_lr = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.9)
    return loss_func, opt, exp_lr


class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes)
            )

        # SE layers
        self.fc1 = nn.Conv1d(planes, planes // 16, kernel_size=1)  # Use nn.Conv1d instead of nn.Linear
        self.fc2 = nn.Conv1d(planes // 16, planes, kernel_size=1)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        # Squeeze
        w = F.avg_pool1d(out, out.size(2))
        w = F.relu(self.fc1(w))
        w = torch.sigmoid(self.fc2(w))
        # Excitation
        out = out * w  # New broadcasting feature from v0.2!

        out += self.shortcut(x)
        out = F.relu(out)
        return out


class PreActBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm1d(in_planes)
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_planes, planes, kernel_size=1, stride=stride, bias=False)
            )

        # SE layers
        self.fc1 = nn.Conv1d(planes, planes // 16, kernel_size=1)
        self.fc2 = nn.Conv1d(planes // 16, planes, kernel_size=1)

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))

        # Squeeze
        w = F.avg_pool1d(out, out.size(2))
        w = F.relu(self.fc1(w))
        w = torch.sigmoid(self.fc2(w))
        # Excitation
        out = out * w

        out += shortcut
        return out


class SENet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=labelNum):
        super(SENet, self).__init__()
        self.model_type = "general"  # general 或者 seq2seq
        self.in_planes = 64

        self.conv1 = nn.Conv1d(featurenum, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # out = F.avg_pool1d(out, 4)
        out = F.adaptive_avg_pool1d(out, 1)  # 改成自适应
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def SENet18():
    return SENet(PreActBlock, [2, 2, 2, 2])


def test():
    net = SENet18()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())


# 做兼容
def Net():
    return SENet18()
