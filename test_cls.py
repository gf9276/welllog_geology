import os
import sys
from multiprocessing import cpu_count

import h5py

curPath = os.path.abspath(os.path.dirname(__file__))  # 加入当前路径，直接执行有用
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
import argparse
from importlib import import_module
import json
import torch

from utils_cls import set_seeds, sample_to_device, save_fig
from dataloader_cls import setup_dataloaders


def parse_args():
    parser = argparse.ArgumentParser(description='Test a model')
    # 文件和路径相关
    parser.add_argument('--config', default='SENet_Geology', help='模型配置文件路径')
    parser.add_argument('--logging_filepath', default='./Log/logging_test.json', help='保存结果的json')
    parser.add_argument('--test_filepath', default='./Data/test.h5', help='测试集路径')
    parser.add_argument('--checkpoint', default='./Log/output.pth', help='模型权重路径')
    # 一些开关
    parser.add_argument('--draw_plt', default="False", help='是否绘图')
    # 其他
    parser.add_argument('--gpu_id', default="0", help='GPU设备')

    args = parser.parse_args()
    return args


def eval_test(net, test_loader, criterion):
    net.eval()  # 很重要
    net.training = False
    total_loss = []
    label_nbr = 0
    eq_nbr = 0
    cur_device = next(net.parameters()).device
    all_label = []
    all_predicted = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            this_batch = sample_to_device(batch, cur_device)
            features = torch.swapaxes(this_batch["features"], 1, 2)
            label = this_batch["label"].long()
            label_nbr += len(label)  # 这是考虑到整体数据量不能被batch size整除

            output = net(features)
            loss = criterion(output, label)

            _, predicted = output.max(1)  # 获取标签
            eq_nbr += predicted.eq(label).sum().item()  # 记录标签准确数量
            total_loss.append(loss.item())  # 记录损失函数

            all_label.append(label)
            all_predicted.append(predicted)

        all_label = torch.cat(tuple(all_label), dim=0)
        all_predicted = torch.cat(tuple(all_predicted), dim=0)

    return eq_nbr / label_nbr, sum(total_loss) / len(total_loss), all_label, all_predicted


def infer(net, test_loader):
    net.eval()  # 很重要
    net.training = False
    cur_device = next(net.parameters()).device
    all_predicted = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            this_batch = sample_to_device(batch, cur_device)
            features = torch.swapaxes(this_batch["features"], 1, 2)
            output = net(features)
            _, predicted = output.max(1)  # 获取标签
            all_predicted.append(predicted)

        all_predicted = torch.cat(tuple(all_predicted), dim=0)

    return all_predicted


def main(args):
    # --------------------O(∩_∩)O-------------- 成功第一步，固定随机数种子 ----------------------------------
    set_seeds()
    # --------------------O(∩_∩)O-------------- 成功第二步，配置参数 ----------------------------------
    cfg = args.config
    logging_filepath = args.logging_filepath
    test_filepath = args.test_filepath
    checkpoint = args.checkpoint
    draw_plt = args.draw_plt
    gpu_id = args.gpu_id
    device = torch.device('cuda:' + gpu_id if torch.cuda.is_available() else 'cpu')
    # 导入模型文件
    x = import_module(cfg)
    batch_size = x.batchsize
    label_classes = x.label

    # --------------------O(∩_∩)O-------------- 成功第四步，加载模型 ----------------------------------
    net = x.Net()
    pretrained_filepath = checkpoint
    loaded_model = torch.load(pretrained_filepath, map_location=torch.device("cpu"))
    net_dict = net.state_dict()

    # 直接判断同名model的尺寸是否一样就可以了，不一样的不加载
    pretrained_dict = {k: v for k, v in loaded_model.items() if k in net_dict and net_dict[k].shape == v.shape}

    net_dict.update(pretrained_dict)  # 更新一下。。
    net.load_state_dict(net_dict, strict=False)

    net = net.to(device)
    criterion, optimizer, exp_lr = x.loss_and_opt(net)
    # --------------------O(∩_∩)O-------------- 成功第五步，训练加验证 ----------------------------------
    print('---------------- 测试 ----------------')
    wells_name = []
    features_h5file = h5py.File(test_filepath, "r")
    for well_name in features_h5file.keys():
        wells_name.append(well_name)
    features_h5file.close()

    result_dict = {}
    for well_name in wells_name:
        test_dataset, test_loader = setup_dataloaders(
            test_filepath,
            label_classes,
            batch_size * 8,  # 我已经全速前进了
            num_workers=cpu_count(),
            shuffle=False,
            which_wells=[well_name])
        if test_dataset.have_label:
            # 有标签就算一下准确率
            test_acc, test_loss, all_label, all_predicted = eval_test(net, test_loader, criterion)
            all_label = test_dataset.reverse_transform_label(all_label)
            all_predicted = test_dataset.reverse_transform_label(all_predicted)
            print("well name: {}, acc = {} / {} = {:.4f}".format(well_name,
                                                                 all_label.eq(all_predicted).sum(),
                                                                 len(all_label),
                                                                 all_label.eq(all_predicted).sum() / len(all_label)))
        else:
            # 没有标签的话就算了
            all_predicted = infer(net, test_loader)
            all_predicted = test_dataset.reverse_transform_label(all_predicted)
            print("well name: {} --> 预测完成".format(well_name))

        result_dict[well_name] = [str(x) for x in all_predicted.cpu().numpy().tolist()]  # 转成string保存
        with open(logging_filepath, "w") as f:
            json.dump(result_dict, f, indent=2)


if __name__ == '__main__':
    # 不要把东西都写到这里面，容易造成全局变量泛滥
    # 在 if __name__ == '__main__': 里面写一长串代码和裸奔没区别
    my_args = parse_args()
    main(my_args)
