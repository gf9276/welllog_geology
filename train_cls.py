import os
import sys
import time
from multiprocessing import cpu_count

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
    """
    提供训练参数
    :return:
    """
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('--config', default="SENet_Geology", help='模型的名字')
    parser.add_argument('--work_dir', default="./Log/logging.json", help='保存日志的位置')
    parser.add_argument('--pretrained', default="True", help='是否要预训练')
    parser.add_argument('--draw_plt', default="False", help='是否绘图')

    args = parser.parse_args()
    return args


def run_epoch(net, train_loader, optimizer, criterion):
    net.train()  # 很重要
    total_loss = 0
    all_nbr = 0
    correct_nbr = 0
    n_batches = len(train_loader)
    cur_device = next(net.parameters()).device
    print_period = 20

    for batch_idx, batch in enumerate(train_loader):
        start = time.time()
        cur_batch_size = len(batch["features"])
        if net.model_type == "general":
            batch = sample_to_device(batch, cur_device)
            features = torch.swapaxes(batch["features"], 1, 2)
            label = batch["label"].long()
            all_nbr += len(label)  # 这是考虑到整体数据量不能被batch size整除

            net.zero_grad()
            optimizer.zero_grad()
            output = net(features)
            loss = criterion(output, label.long())
            loss.backward()
            optimizer.step()

            _, predicted = output.max(1)  # 获取标签
            correct_nbr += predicted.eq(label).sum().item()  # 记录标签准确数量
            total_loss += loss.item()  # 记录损失函数
        elif net.model_type == "transformer":
            # batch = sample_to_device(batch, cur_device)
            # features = batch["features"]
            # multi_label = batch["multi_label"].long()
            # transformer_batch = net.get_batch(features, None, 926219)
            # all_nbr += len(multi_label.contiguous().view(-1))  # 这是考虑到整体数据量不能被batch size整除
            #
            # net.zero_grad()
            # optimizer.zero_grad()
            # output = net.forward(src=transformer_batch.src, src_mask=transformer_batch.src_mask, just_encoder=True)
            # loss = criterion(output.contiguous().view(-1, output.size(-1)), multi_label.contiguous().view(-1))
            # loss.backward()
            # optimizer.step()
            #
            # _, predicted = output.contiguous().view(-1, output.size(-1)).max(1)  # 获取标签
            # correct_nbr += predicted.eq(multi_label.contiguous().view(-1)).sum().item()  # 记录标签准确数量
            # total_loss += loss.item()  # 记录损失函数

            batch = sample_to_device(batch, cur_device)
            features = batch["features"]
            transformer_batch = net.get_batch(features, None, 926219)
            label = batch["label"].long()
            all_nbr += len(label)  # 这是考虑到整体数据量不能被batch size整除

            net.zero_grad()
            optimizer.zero_grad()
            output = net.forward(src=transformer_batch.src, src_mask=transformer_batch.src_mask, just_encoder=True)
            output = output[:, output.shape[1] // 2, :]
            loss = criterion(output, label.long())
            loss.backward()
            optimizer.step()

            _, predicted = output.max(1)  # 获取标签
            correct_nbr += predicted.eq(label).sum().item()  # 记录标签准确数量
            total_loss += loss.item()  # 记录损失函数

            if batch_idx % print_period == 0:
                use_time = time.time() - start
                print("Epoch Step: %d, Loss: %f , Acc: %f, Tokens per Sec: %f" %
                      (batch_idx, total_loss / (batch_idx + 1), correct_nbr / all_nbr, use_time / cur_batch_size))

    return correct_nbr / all_nbr, total_loss / n_batches


def eval_val(net, val_loader, criterion):
    total_loss = 0
    all_nbr = 0
    correct_nbr = 0
    net.eval()  # 很重要
    net.training = False
    n_batches = len(val_loader)
    cur_device = next(net.parameters()).device
    total_output = []
    total_label = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if net.model_type == "general":
                batch = sample_to_device(batch, cur_device)
                features = torch.swapaxes(batch["features"], 1, 2)
                label = batch["label"].long()
                all_nbr += len(label)  # 这是考虑到整体数据量不能被batch size整除

                output = net(features)
                loss = criterion(output, label.long())

                _, predicted = output.max(1)  # 获取标签

                total_output += output
                total_label += label.squeeze()

                correct_nbr += predicted.eq(label).sum().item()  # 记录标签准确数量
                total_loss += loss.item()  # 记录损失函数
            elif net.model_type == "transformer":
                batch = sample_to_device(batch, cur_device)
                features = batch["features"]
                multi_label = batch["multi_label"].long()
                transformer_batch = net.get_batch(features, None, 926219)
                all_nbr += len(multi_label.contiguous().view(-1))  # 这是考虑到整体数据量不能被batch size整除

                output = net.forward(src=transformer_batch.src, src_mask=transformer_batch.src_mask, just_encoder=True)
                loss = criterion(output.contiguous().view(-1, output.size(-1)), multi_label.contiguous().view(-1))

                _, predicted = output.contiguous().view(-1, output.size(-1)).max(1)  # 获取标签
                predicted = val_loader.dataset.remove_overlap(predicted)
                label = val_loader.dataset.remove_overlap(multi_label.contiguous().view(-1))
                correct_nbr += predicted.eq(label).sum().item()  # 记录标签准确数量
                total_loss += loss.item()  # 记录损失函数
                # batch = sample_to_device(batch, cur_device)
                # features = batch["features"]
                # label = batch["label"].long()
                # transformer_batch = net.get_batch(features, None, 926219)
                # all_nbr += len(label)  # 这是考虑到整体数据量不能被batch size整除
                #
                # output = net.forward(src=transformer_batch.src, src_mask=transformer_batch.src_mask, just_encoder=True)
                # output = output[:, output.shape[1] // 2, :]
                # loss = criterion(output, label.long())
                #
                # _, predicted = output.max(1)  # 获取标签
                # correct_nbr += predicted.eq(label).sum().item()  # 记录标签准确数量
                # total_loss += loss.item()  # 记录损失函数

    return correct_nbr / all_nbr, total_loss / n_batches, total_output, total_label


def main(args):
    # --------------------O(∩_∩)O-------------- 成功第一步，固定随机数种子 ----------------------------------
    set_seeds(9276)
    # --------------------O(∩_∩)O-------------- 成功第二步，配置参数 ----------------------------------
    cfg = args.config
    work_dir = args.work_dir
    pretrained = args.pretrained
    draw_plt = args.draw_plt
    # 导入模型文件
    x = import_module(cfg)
    # 获取文件路径
    train_h5filepath = x.trainDir
    val_h5filepath = x.valDir
    train_label_filepath = x.trainLabel
    val_label_filepath = x.valLabel
    model_save_path = x.savePath
    # 获取训练的一些参数
    epoch_nbr = x.epoch
    gpu_id = str(x.gpuId)
    batch_size = x.batchsize
    device = torch.device('cuda:' + gpu_id if torch.cuda.is_available() else 'cpu')
    label_nbr = x.labelNum
    label_classes = x.label
    features_nbr = x.featurenum
    slice_length = x.slicelength
    # --------------------O(∩_∩)O-------------- 成功第三步，加载数据集 ----------------------------------
    train_dataset, train_loader = setup_dataloaders(
        train_h5filepath,
        train_label_filepath,
        label_classes,
        batch_size,
        num_workers=cpu_count(),
        shuffle=True,
        noise=True)
    val_dataset, val_loader = setup_dataloaders(
        val_h5filepath,
        val_label_filepath,
        label_classes,
        batch_size,
        num_workers=cpu_count(),
        shuffle=False)

    # --------------------O(∩_∩)O-------------- 成功第四步，加载模型 ----------------------------------
    net = x.Net()
    if pretrained == "True":
        pretrained_filepath = x.pretrainedPath
        loaded_model = torch.load(pretrained_filepath)
        net_dict = net.state_dict()

        # 直接判断同名model的尺寸是否一样就可以了，不一样的不加载
        pretrained_dict = {k: v for k, v in loaded_model.items() if k in net_dict and net_dict[k].shape == v.shape}

        net_dict.update(pretrained_dict)  # 更新一下。。
        net.load_state_dict(net_dict, strict=False)

    net = net.to(device)
    criterion, optimizer, exp_lr = x.loss_and_opt(net)
    # --------------------O(∩_∩)O-------------- 成功第五步，训练加验证 ----------------------------------
    result_dict = {"trainloss": [], "valloss": [], "trainaccuracy": [], "valaccuracy": []}  # 用于保存记录到json里面的字典
    print('---------------- 训练前测试一波 ----------------')
    val_acc, val_loss, _, _ = eval_val(net, val_loader, criterion)
    best_acc = val_acc
    best_epoch = -1
    print("测试集准确率: {:.2f}%, 测试集loss: {:.4f}".format(100 * val_acc, val_loss))
    for cur_epoch in range(epoch_nbr):
        print('---------------- 当前epoch: {} ----------------'.format(cur_epoch))
        print('当前lr:', optimizer.state_dict()['param_groups'][0]['lr'])
        # 训练一次
        train_acc, train_loss = run_epoch(net, train_loader, optimizer, criterion)
        exp_lr.step()  # 学习率迭代

        # 验证一次
        val_acc, val_loss, total_output, total_label = eval_val(net, val_loader, criterion)

        # 记录结果，并打印
        result_dict['trainaccuracy'].append(train_acc)
        result_dict['valaccuracy'].append(val_acc)
        result_dict['trainloss'].append(train_loss)
        result_dict['valloss'].append(val_loss)
        print("训练集准确率: {:.4f}%, 测试集准确率: {:.4f}%, 训练集loss: {:.4f}, 测试集loss: {:.4f}"
              .format(100 * train_acc, 100 * val_acc, train_loss, val_loss))

        # 保存验证集表现最好的模型
        if best_acc < val_acc:
            best_acc = val_acc
            best_epoch = cur_epoch
            print('best acc: {:.4f}%, best epoch: {}'.format(100 * best_acc, best_epoch))
            print('save model')
            torch.save(net.state_dict(), model_save_path)  # 转到cpu，再保存参数

        # 保存json和图片
        with open(work_dir, "w") as f:
            json.dump(result_dict, f, indent=2)

        if draw_plt == "True":
            save_fig(result_dict['trainaccuracy'], "accuracy of train set", './Log/train_acc.png')
            save_fig(result_dict['valaccuracy'], "accuracy of val set", './Log/val_acc.png')
            save_fig(result_dict['trainloss'], "train loss", './Log/train_loss.png')
            save_fig(result_dict['valloss'], "val loss", './Log/val_loss.png')

    # --------------------O(∩_∩)O-------------- 成功第六步，结束 ----------------------------------
    print('---------------- 训练结束，完结撒花 ----------------')
    print('best acc: {:.4f}%, best epoch: {}'.format(100 * best_acc, best_epoch))


if __name__ == '__main__':
    # 不要把东西都写到这里面，容易造成全局变量泛滥
    # 在 if __name__ == '__main__': 里面写一长串代码和裸奔没区别
    my_args = parse_args()
    main(my_args)
