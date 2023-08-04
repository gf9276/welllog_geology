import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# training config
slicelength = 97
epoch = 100
batchsize = 1024
learningrate = 0.001
lossfunc = 0

# network config
featurenum = 6
label = ['34', '19', '3', '4', '25', '2', '1', '0', '23', '35', '32', '33', '31', '30', '28', '29', '26', '27', '22', '21', '12', '13', '14', '15', '16', '17',
         '18', '24', '11', '20', '5', '6', '7', '8', '9', '10']

# -------- Do not delete this line, the configuration ends here. --------

# --------- 变量映射 & 二次处理 & 通用函数定义, 我不喜欢原先的变量命名. ---------
model_name = 'Transformer'
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


def Net():
    return make_model(features_num, label_num, d_model=512)


class EncoderDecoder(nn.Module):
    """
    标准的Encoder-Decoder架构。这是很多模型的基础
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        """
        生成编码解码器
        :param encoder: 是个函数
        :param decoder: 是个函数
        :param src_embed: 是个函数
        :param tgt_embed: 是个函数
        :param generator: 是个函数，这里里面没有用到，外面应该会调用
        """
        super(EncoderDecoder, self).__init__()
        self.model_type = "transformer"
        # encoder和decoder都是构造的时候传入的，这样会非常灵活
        self.encoder = encoder
        self.decoder = decoder
        # 源语言和目标语言的embedding
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        # generator后面会讲到，就是根据Decoder的隐状态输出当前时刻的词
        # 基本的实现就是隐状态输入一个全连接层，全连接层的输出大小是词的个数
        # 然后接一个softmax变成概率。
        self.generator = generator
        self.get_batch = Batch

    def forward(self, src, src_mask, tgt=None, tgt_mask=None, just_encoder=True):
        # 首先调用encode方法对输入进行编码，然后调用decode方法解码
        if just_encoder:
            return self.generator(self.encode(src, src_mask))
        return self.generator(self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask))

    def encode(self, src, src_mask):
        # 调用encoder来进行编码，传入的参数embedding的src和src_mask
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        # 调用decoder
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


class Generator(nn.Module):
    # 根据Decoder的隐状态输出一个词
    # d_model是Decoder输出的大小，vocab是词典大小
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    # 全连接再加上一个softmax
    def forward(self, x):
        # 在softmax的结果上再做多一次log运算，em mmmm，外面应该会调用NLLLoss()
        x = self.proj(x)
        # return F.log_softmax(x, dim=-1)
        # return F.log_softmax(x[:, :, :-1], dim=-1)
        return x


def clones(module, n):
    """
    克隆N个完全相同的SubLayer，使用了copy.deepcopy
    :param module: 模型
    :param n: n个
    :return:
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


class Encoder(nn.Module):
    """
    Encoder是N个EncoderLayer的stack
    """

    def __init__(self, layer, n):
        """
        构建encoder
        :param layer:
        :param n:
        """
        super(Encoder, self).__init__()
        # layer是一个SubLayer，我们clone N个
        self.layers = clones(layer, n)
        # 再加一个LayerNorm层
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        """
        逐层进行处理
        :param x:
        :param mask:
        :return:
        """
        for layer in self.layers:
            x = layer(x, mask)
        # 最后进行LayerNorm，后面会解释为什么最后还有一个LayerNorm。
        return self.norm(x)


class LayerNorm(nn.Module):
    """
    一种归一化方法，替代batchNorm的
    """

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    LayerNorm + sublayer(Self-Attenion/Dense) + dropout + 残差连接
    为了简单，把LayerNorm放到了前面，这和原始论文稍有不同，原始论文LayerNorm在最后。
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """
        sublayer是传入的参数，参考DecoderLayer，它可以当成函数调用，这个函数的有一个输入参数
        :param x:
        :param sublayer:
        :return:
        """
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    """
    EncoderLayer由self-attn和feed forward组成
    """

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        """
        Follow Figure 1 (left) for connections.
        :param x:
        :param mask:
        :return:
        """
        z = lambda y: self.self_attn(y, y, y, mask)
        x = self.sublayer[0](x, z)
        return self.sublayer[1](x, self.feed_forward)


class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    """
    Decoder包括self-attn, src-attn, 和feed forward
    """

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


def get_subsequent_mask(size):
    """
    Mask out subsequent positions.
    获取子序列掩码
    :param size:
    :return:
    """
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # 所有h个head的mask都是相同的
            mask = mask.unsqueeze(1)
        n_batches = query.size(0)

        # 1) 首先使用线性变换，然后把d_model分配给h个Head，每个head为d_k=d_model/h
        query, key, value = [l(x).view(n_batches, -1, self.h, self.d_k).transpose(1, 2) for l, x in zip(self.linears, (query, key, value))]

        # 2) 使用attention函数计算
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) 把8个head的64维向量拼接成一个512的向量。然后再使用一个线性变换(512,521)，shape不变。
        x = x.transpose(1, 2).contiguous().view(n_batches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class EmbeddingsLiner(nn.Module):
    """
    其实就是简单的转换一下尺寸。。。转换一下词向量的维度
    """

    def __init__(self, h_in, h_out):
        super(EmbeddingsLiner, self).__init__()
        self.liner = nn.Linear(h_in, h_out)
        self.h_out = h_out

    def forward(self, x):
        # return self.liner(x) * math.sqrt(self.h_out)  # 这里为什么要乘以根号下的h_out呢
        return F.normalize(self.liner(x), 2, dim=2) * math.sqrt(self.h_out)  # L2归一化一下
        # return self.liner(x)  # L2归一化一下
        # (F.normalize(self.liner(x),2,dim=2)[1,1]**2).sum()


class PositionalEncoding(nn.Module):
    """Implement the PE function."""

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)


def make_model(h_in, tgt_vocab, n=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    """Helper: Construct a model from hyperparameters."""
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), n),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), n),
        nn.Sequential(EmbeddingsLiner(h_in, d_model), c(position)),  # 这里被我改成了维度转换
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab),
    )

    # 随机初始化参数，这非常重要
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


class Batch:
    """
    Object for holding a batch of data with mask during training.
    """

    def __init__(self, src, tgt=None, pad=2):  # 2 = <blank>
        # self.src = src
        # self.src_mask = (src != pad).unsqueeze(-2)
        self.src = src
        self.src_mask = (src[:, :, 0] != pad).unsqueeze(-2)  # 没啥用，因为我都是同样大小的，所以这里设置的都是完整的
        if tgt is not None:
            self.tgt = tgt[:, :-1]
            self.tgt_y = tgt[:, 1:]
            self.tgt_mask = self.make_std_mask(self.tgt, pad)
            self.n_tokens = (self.tgt_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        """创建Mask，使得我们不能attend to未来的词"""
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & get_subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data)
        return tgt_mask


class SimpleLossCompute:
    """
    A simple loss compute and train function.
    """

    def __init__(self, generator, criterion):
        self.generator = generator  # 前面的生成器，输出应该是F.log_softmax
        self.criterion = criterion  # 损失函数

    def __call__(self, x, y, norm):
        x = self.generator(x)
        loss = (self.criterion(x.contiguous().view(-1, x.size(-1)), y.contiguous().view(-1)) / norm)
        return loss.data * norm, loss
