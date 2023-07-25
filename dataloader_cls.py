import random

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


class MyDataSet(Dataset):
    def __init__(self, h5filepath: str, label_filepath: str, label_classes: list, which_wells=None, noise=False):
        """
        label_classes_dict: 比如 {'3': 0, '5': 1, '1': 2, '0': 3, '6': 4, '4': 5, '2': 6, '-99999': 7}
        """

        # a-->b b-->a
        self.label_classes_dict = dict(zip(label_classes, list(range(len(label_classes)))))
        self.label_classes_reversal_dict = dict(zip(list(range(len(label_classes))), label_classes))
        self.h5filepath = h5filepath
        self.label_filepath = label_filepath
        self.noise = noise
        self.have_label = True

        self.label_classes_nbr = len(self.label_classes_dict)  # 标签数量

        # 直接全部加载到内存里，其实可以在__getitem__里面重新打开文件，会不会起冲突我就不知道了，没试过
        self.dataset = {}
        self.wells_name = []  # 保存井次信息 --> 必须list，python里的dict貌似没有顺序，新版本可能有
        self.wells_size = []  # 保存每个wells对应的长度

        self.features_h5file = h5py.File(h5filepath, "r")
        self.slice_length = self.features_h5file.attrs["slice_length"]
        self.slice_step = self.features_h5file.attrs["slice_step"]
        all_well_features = []
        all_well_label = []
        all_well_multi_label = []
        for well_name in self.features_h5file.keys():
            if which_wells is not None and well_name not in which_wells:
                continue
            self.wells_name.append(well_name)
            self.wells_size.append(len(self.features_h5file[well_name]["features"][:]))

            # 按照float32的格式读取
            all_well_features.append(self.features_h5file[well_name]["features"][:].astype("float32"))
            if "label" in self.features_h5file[well_name].keys() and "multi_label" in self.features_h5file[well_name].keys():
                # 四舍六入五凑偶，因为我在处理数据集的时候，为了兼容“参数估计”任务，保存的标签也是float
                all_well_label.append(np.round(self.features_h5file[well_name]["label"][:]).astype("int32"))
                all_well_multi_label.append(np.round(self.features_h5file[well_name]["multi_label"][:]).astype("int32"))
            else:
                self.have_label = False

        self.features_h5file.close()

        self.dataset["features"] = np.concatenate(tuple(all_well_features), 0)
        if self.have_label:
            self.dataset["label"] = self.transform_label(np.concatenate(tuple(all_well_label), 0))
            self.dataset["multi_label"] = self.transform_label(np.concatenate(tuple(all_well_multi_label), 0))

        self.features_nbr = self.dataset["features"].shape[2]
        self.slice_length = self.dataset["features"].shape[1]

        self.features_mean = np.mean(self.dataset["features"].reshape(-1, self.features_nbr), axis=0)
        self.features_std = np.std(self.dataset["features"].reshape(-1, self.features_nbr), axis=0)
        pass

        # xxx = self.remove_overlap(torch.as_tensor(self.dataset["multi_label"]).contiguous().view(-1).long())
        # yyy = self.reverse_transform_label(xxx)

    def __len__(self):
        # 返回数据集长度
        return len(self.dataset["features"])

    def __getitem__(self, idx):
        """
        原来聪明的dataloader，会自动转化为tensor，那我就不多此一举了
        """
        features = np.copy(self.dataset["features"][idx])
        if self.noise:
            # 貌似没什么效果
            for i in range(self.features_nbr):
                noise = 0.1 * np.random.normal(loc=self.features_mean[i], scale=self.features_std[i], size=features[:, i].shape)
                features[:, i] += noise

        if self.have_label:
            label = np.copy(self.dataset["label"][idx]).squeeze()
            multi_label = np.copy(self.dataset["multi_label"][idx]).squeeze()
            return {"features": features, "label": label, "multi_label": multi_label, **self.get_data_well(idx)}
        else:
            return {"features": features, **self.get_data_well(idx)}

    def get_data_well(self, idx):
        """
        解析获得数据对应的井次和井次里的idx
        其实这个用处不大
        :return:
        """
        output = {}
        well_idx = idx
        for i in range(len(self.wells_name)):
            if well_idx - self.wells_size[i] < 0:
                output = {'well': self.wells_name[i], "idx": well_idx}
                break
            else:
                well_idx -= self.wells_size[i]

        return output

    def transform_label(self, label):
        """
        将标签转换一下罢了
        """
        mask = {}
        for key in self.label_classes_dict.keys():
            mask[key] = label == int(key)
        # 再根据mask填充
        for key in self.label_classes_dict.keys():
            label[mask[key]] = self.label_classes_dict[key]  # 解释标签需要转换成 0 1 2 3 4 5 6 7 这样子

        return label

    def reverse_transform_label(self, label):
        """
        将标签反方向转换一下罢了
        """
        mask = {}
        for key in self.label_classes_reversal_dict.keys():
            mask[key] = label == int(key)
        # 再根据mask填充
        for key in self.label_classes_reversal_dict.keys():
            label[mask[key]] = int(self.label_classes_reversal_dict[key])

        return label

    def remove_overlap(self, total_label):
        """
        """
        output = []

        # 针对每一个井，进行一次标签解析，这样可以防止交叉处的问题
        last_idx = 0
        for i in range(len(self.wells_size)):
            cur_well_total_label = total_label[last_idx:last_idx + self.wells_size[i] * self.slice_length]
            output.append(self.remove_overlap_in_well(cur_well_total_label))
            last_idx = last_idx + self.wells_size[i] * self.slice_length

        return torch.cat(tuple(output), dim=0)

    def remove_overlap_in_well(self, total_label):
        """
        total_label：n, 97 -- > 拼接成n*97 后的内容
        必须是同一口井，必须是按照深度顺序的
        :return:
        """
        slice_nbr = int(len(total_label) / self.slice_length)  # 切片数，其实就是数据集的数据
        depth_point_nbr = int(self.slice_step * slice_nbr + self.slice_length)  # 深度点个数就是切片数量+最后面的切片

        voting_matrix = torch.zeros([depth_point_nbr, self.label_classes_nbr], dtype=torch.float32)  # 准备开始计数，投票矩阵
        voting_matrix = voting_matrix.to(total_label.device)

        for i in range(slice_nbr):
            cur_label = total_label[i * self.slice_length:(i + 1) * self.slice_length]  # 当前的长度为97的label

            score = voting_matrix[i * self.slice_step:i * self.slice_step + self.slice_length].gather(1, cur_label.unsqueeze(1))
            score += 1

            # scatter_就是inplace，而scatter则不会
            voting_matrix[i * self.slice_step:i * self.slice_step + self.slice_length].scatter_(1, cur_label.unsqueeze(1), score)
            # voting_matrix[i * self.slice_step:i * self.slice_step + self.slice_length] = torch.scatter(
            #     voting_matrix[i * self.slice_step:i * self.slice_step + self.slice_length], 1, cur_label.unsqueeze(1), score)

            pass  # 用来打断点的

        output = voting_matrix.argmax(1)
        return output


def setup_dataloaders(h5filepath: str,
                      label_filepath: str,
                      label_classes: list,
                      batch_size: int,
                      num_workers=0,
                      shuffle=True,
                      which_wells=None,
                      noise=False):
    """
    Prepare datasets for training, validation and test.
    """

    def _worker_init_fn(worker_id):
        """
        Worker init fn to fix the seed of the workers
        用来固定数据加载过程中的线程随机数种子的
        """

        seed = torch.initial_seed() % 2 ** 32 + worker_id  # worker_id 可以不加，每个epoch都不一样，**优先级很高的
        np.random.seed(seed)
        random.seed(seed)

    dataset = MyDataSet(h5filepath, label_filepath, label_classes, which_wells, noise)
    sampler = None  # 这个是在多GPU上用的，我没搞多GPU
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=shuffle,
        num_workers=num_workers,
        worker_init_fn=_worker_init_fn,
        sampler=sampler,
        drop_last=False
    )

    return dataset, loader  # 数据集获取成功，外面一般用的都是 train_loader
