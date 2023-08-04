import numpy as np
import os
import random
import torch
from enum import Enum


class Bcolors(Enum):
    """
    和 shell 一样的
    """
    HEADER = '\033[95m'  # 紫色
    OKBLUE = '\033[94m'  # 蓝色 --> 两个交替着来
    OKGREEN = '\033[92m'  # 绿色 --> 两个交替着来
    WARNING = '\033[93m'  # 黄色 --> 警告
    TIPS = '\033[96m'  # 青色 --> 提示(这个太亮了)
    FAIL = '\033[91m'  # 深红 --> 失败
    ENDC = '\033[0m'  # 关闭所有属性，即属性结束标识
    BOLD = '\033[1m'  # 设置高亮度，即字体加粗、文体强调
    UNDERLINE = '\033[4m'  # 下划线


# echo -e "\e[90m 黑底黑字 \e[0m"
# echo -e "\e[91m 黑底红字 \e[0m"
# echo -e "\e[92m 黑底绿字 \e[0m"
# echo -e "\e[93m 黑底黄字 \e[0m"
# echo -e "\e[94m 黑底蓝字 \e[0m"
# echo -e "\e[95m 黑底紫字 \e[0m"
# echo -e "\e[96m 黑底青字 \e[0m"
# echo -e "\e[97m 黑底白字 \e[0m"

def printcolor(message, color=Bcolors.ENDC):
    """
    Print a message in a certain color (only rank 0)
    grey, red, green, yellow, blue, magenta, cyan, white.
    """

    # print(colored(message, color))
    # 因为 termcolor 没用，所以我自己弄了个
    if color not in Bcolors:
        color = Bcolors.ENDC
    print(color.value + str(message) + Bcolors.ENDC.value)


def print_block(message, title='', color=Bcolors.ENDC):
    """
    打印区块信息
    :param message:
    :param title:
    :param color:
    :return:
    """
    print('')
    printcolor('-' * 25 + ' ' + title + ' ' + '-' * 25, color)
    printcolor(message, color)
    print('')


def set_seeds(seed=43):
    """
    Set Python random seeding and PyTorch seeds.
    固定随机数种子

    Parameters
    ----------
    seed: int, default: 42
        Random number generator seeds for PyTorch and python
    """
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def sample_to_device(data, device="cpu"):
    """
    将数据 放到cuda 里
    """
    if isinstance(data, str):
        return data
    if isinstance(data, dict):
        data_cuda = {}
        for key in data.keys():
            data_cuda[key] = sample_to_device(data[key], device)
        return data_cuda
    elif isinstance(data, list):
        data_cuda = []
        for key in data:
            data_cuda.append(sample_to_device(key, device))
        return data_cuda
    else:
        return data.to(device)


def save_fig(values, title, fig_path):
    # 使用agg而不是默认的qt5agg
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    t = np.arange(0, len(values))
    plt.plot(t, values)
    plt.title(title)
    plt.savefig(fig_path)
    plt.close()
