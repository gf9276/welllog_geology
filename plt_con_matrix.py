import numpy as np
import torch
from matplotlib import pyplot as plt

# 只更改这里的标签即可
label_total = ['36', '21', '3', '4', '27', '2', '1', '0', '25', '37', '34', '35', '33', '32', '30', '31', '28', '29', '24', '23', '13', '14', '15', '16', '17',
               '18', '19', '26', '12', '22', '5', '6', '20', '7', '8', '9', '10', '11']

num_label = len(label_total)


def plot_conf(output, label, epoch):
    # output=sum(output.tolist(),[])
    # label=sum(label.tolist(),[])

    conf_matrix = make_confusion_matrix(output, label)

    corrects = conf_matrix.diagonal(offset=0)  # 抽取对角线的每种分类的识别正确个数
    per_kinds = conf_matrix.sum(axis=1)  # 抽取每个分类数据总的测试条数
    # print("混淆矩阵总元素个数：{0}".format(int(np.sum(conf_matrix))))
    # print(conf_matrix)
    # 获取每种Emotion的识别准确率
    # print("每种类别",per_kinds)
    # print("每种类别预测正确的个数：",corrects)
    # print("每种类别的识别准确率为：{0}".format([rate*100 for rate in corrects/per_kinds]))
    plt.figure(figsize=(10, 10))
    plt.imshow(conf_matrix, cmap=plt.cm.Blues)
    font_dict = dict(fontsize=8,
                     color='b',

                     weight='light',
                     style='italic',
                     )
    # 在图中标注数量/概率信息
    thresh = conf_matrix.max() / 2  # 数值颜色阈值，如果数值超过这个，就颜色加深。
    for x in range(num_label):
        for y in range(num_label):
            # 注意这里的matrix[y, x]不是matrix[x, y]

            info = int(conf_matrix[y, x])

            if info != 0:
                plt.text(x, y, info,
                         font_dict,
                         verticalalignment='center',
                         horizontalalignment='center',
                         color="white" if info > thresh else "black"
                         )
    # embed()
    plt.tight_layout()  # 保(证图不)重叠
    plt.yticks(range(num_label), label_total)
    # plt.xticks(range(num_label), label,rotation=45)#X轴字体倾斜45°
    plt.xticks(range(num_label), label_total, rotation=45)
    plt.savefig('confusion_matrix_epoch.png')
    plt.close()


def make_confusion_matrix(preds, labels):
    conf_matrix = torch.zeros(num_label, num_label)
    for pred, lab in zip(preds, labels):
        pred = torch.argmax(pred, 1).cpu()
        lab = lab.cpu()

        for p, t in zip(pred, lab):
            conf_matrix[p, t] += 1
    conf_matrix = np.array(conf_matrix)
    return conf_matrix
