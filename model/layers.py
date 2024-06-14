import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn import global_add_pool, global_mean_pool
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax, to_dense_batch

from typing import Callable, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.nn import Parameter, Sigmoid

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import zeros
from torch_geometric.typing import Adj, OptTensor, PairTensor
from torch_scatter import scatter_add
from typing import Callable, Optional, Tuple, Union
import torch
from torch import Tensor
from torch.nn import Parameter, Sigmoid
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import zeros
from torch_geometric.typing import Adj, OptTensor, PairTensor

import inspect
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Dropout, Linear, Sequential

from attention.la import *
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import reset
from torch_geometric.nn.resolver import (
    activation_resolver,
    normalization_resolver,
)
from torch_geometric.typing import Adj
from torch_geometric.utils import to_dense_batch
from torch_geometric.data import Batch
import torch
from torch_geometric.nn import GINConv


class MLPModule(torch.nn.Module):
    def __init__(self, nhid):
        super(MLPModule, self).__init__()
        self.nhid = nhid
        self.dropout = 0.0

        self.lin0 = torch.nn.Linear(nhid, self.nhid * 2)
        nn.init.xavier_uniform_(self.lin0.weight.data)
        nn.init.zeros_(self.lin0.bias.data)

        self.lin1 = torch.nn.Linear(self.nhid * 2, self.nhid)
        nn.init.xavier_uniform_(self.lin1.weight.data)
        nn.init.zeros_(self.lin1.bias.data)

        self.lin2 = torch.nn.Linear(self.nhid, self.nhid // 2)
        nn.init.xavier_uniform_(self.lin2.weight.data)
        nn.init.zeros_(self.lin2.bias.data)

        self.lin3 = torch.nn.Linear(self.nhid // 2, 1)
        nn.init.xavier_uniform_(self.lin3.weight.data)
        nn.init.zeros_(self.lin3.bias.data)

    def forward(self, scores):
        scores = self.lin0(scores)
        scores = F.dropout(scores, p=self.dropout, training=self.training)
        scores = F.leaky_relu(self.lin1(scores))
        scores = F.dropout(scores, p=self.dropout, training=self.training)
        scores = F.leaky_relu(self.lin2(scores))
        scores = F.dropout(scores, p=self.dropout, training=self.training)
        scores = torch.sigmoid(self.lin3(scores)).view(-1)

        return scores


class ReadoutModule2(torch.nn.Module):
    def __init__(self, nhid):
        super(ReadoutModule2, self).__init__()
        self.nhid = nhid

        self.weight = torch.nn.Parameter(torch.Tensor(self.nhid, self.nhid))
        nn.init.xavier_uniform_(self.weight.data)

    def forward(self, x, batch):
        mean_pool = global_mean_pool(x, batch)
        transformed_global = torch.tanh(torch.mm(mean_pool, self.weight))
        coefs = torch.sigmoid((x * transformed_global[batch]).sum(dim=1))
        weighted = coefs.unsqueeze(-1) * x
        return global_add_pool(weighted, batch)


import torch.nn.functional as F

import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool

import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool, global_max_pool

from typing import Callable, Optional, Tuple, Union

import torch
from torch import Tensor

from torch_geometric.nn.pool.topk_pool import topk


#
# class ReadoutModule(nn.Module):
#     def __init__(self, nhid):
#         super(ReadoutModule, self).__init__()
#         self.nhid = nhid
#         self.weight1 = nn.Parameter(torch.Tensor(self.nhid, self.nhid))
#         self.weight2 = nn.Parameter(torch.Tensor(self.nhid, self.nhid))
#
#         nn.init.xavier_uniform_(self.weight1.data)
#         nn.init.xavier_uniform_(self.weight2.data)
#
#     def forward(self, x, batch):
#         mean_pool = global_mean_pool(x, batch)
#         max_pool = global_max_pool(x, batch)
#
#         transformed_global_mean = torch.tanh(torch.mm(mean_pool, self.weight1))
#         transformed_global_max = torch.tanh(torch.mm(max_pool, self.weight2))
#
#         coefs_mean = torch.sigmoid((x * transformed_global_mean[batch]).sum(dim=1))
#         coefs_max = torch.sigmoid((x * transformed_global_max[batch]).sum(dim=1))
#
#         weighted_mean = coefs_mean.unsqueeze(-1) * x
#         weighted_max = coefs_max.unsqueeze(-1) * x
#
#         return global_add_pool(weighted_mean, batch), global_add_pool(weighted_max, batch)


# class Combinegraph(nn.Module):
#     def __init__(self, dim_size):
#         super(Combinegraph, self).__init__()
#         self.dim_size = dim_size
#         self.ReadoutModule2 = ReadoutModule(dim_size)
#         self.bi = nn.Bilinear(dim_size, dim_size, dim_size)
#         self.ha = VectorSimilarity(dim_size // 2)
#
#     def forward(self, x1, batch1, x2, batch2):
#         embed1_mean, embed1_max = self.ReadoutModule2(x1, batch1)
#         embed2_mean, embed2_max = self.ReadoutModule2(x2, batch2)
#         score_mean = self.ha.process_vectors(embed1_mean, embed2_mean)
#         score_max = self.bi(embed1_max, embed2_max)
#         scroce = torch.concat((score_mean, score_max), dim=1)
#
#         return scroce

class SETensorNetworkModule(torch.nn.Module):
    def __init__(self, dim_size):
        super(SETensorNetworkModule, self).__init__()

        self.dim_size = dim_size
        self.setup_weights()

    def setup_weights(self):
        channel = self.dim_size * 2
        reduction = 4
        self.fc_se = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

        self.fc0 = nn.Sequential(
            nn.Linear(channel, channel),
            nn.ReLU(inplace=True),
            nn.Linear(channel, channel),
            nn.ReLU(inplace=True)
        )

        self.fc1 = nn.Sequential(
            nn.Linear(channel, channel),
            nn.ReLU(inplace=True),
            nn.Linear(channel, self.dim_size),  # nn.Linear(channel, self.args.tensor_neurons),
            nn.ReLU(inplace=True)
        )

    def forward(self, embedding_1, embedding_2):
        combined_representation = torch.cat((embedding_1, embedding_2), 1)
        se_feat_coefs = self.fc_se(combined_representation)
        se_feat = se_feat_coefs * combined_representation + combined_representation
        scores = self.fc1(se_feat)

        return scores


class Combineall(nn.Module):
    def __init__(self, dim_size, numheads=8):
        super(Combineall, self).__init__()
        self.dim_size = dim_size
        self.numheads = numheads
        self.ReadoutModule = ReadoutModule2(dim_size)
        # self.linear = nn.Linear(dim_size, dim_size * numheads)
        self.ha = VectorSimilarity(dim_size // 2)
        # self.fusion=SETensorNetworkModule(dim_size)
        self.Binode = EnhancedBilinearInteraction(dim_size, dim_size, dim_size)
        # self.Binode_graph = EnhancedBilinearInteraction(dim_size, dim_size, dim_size)
        # self.BiDAFWithAttention = BiDAFWithAttention(dim_size, dim_size // 2)

    def forward(self, x1, batch1, x2, batch2):
        embed1_mean = self.ReadoutModule(x1, batch1)
        embed2_mean = self.ReadoutModule(x2, batch2)
        h1, mask1, h2, mask2, _ = merge_batches_and_process(x1, batch1, x2, batch2)
        scoreh = self.Binode(h1, h2)
        score_mean = self.ha.process_vectors(embed1_mean, embed2_mean)
        return torch.concat((score_mean, scoreh), dim=-1)


class VGS(nn.Module):
    def __init__(self, sigma=2.0):
        super(VGS, self).__init__()
        # sigma squared as a trainable parameter
        # self.sigma_squared = nn.Parameter(torch.tensor([sigma ** 2]))

    def forward(self, v_i, v_j):  # 输入形状为 [B, L, dim]
        # Compute the difference vector
        diff = v_i - v_j
        # Compute the squared Euclidean distance
        distance_squared = torch.sum(diff ** 2, dim=-1)
        # Compute the similarity score using the Gaussian function
        similarity = torch.exp(-distance_squared / 4)
        return similarity


class VectorSimilarity:
    def __init__(self, max_poolings, head=4):
        self.max_poolings = max_poolings
        self.kernels = [1, 2, 3, 4, 5, 6, 7, 8]
        self.strides = [1, 2, 3]
        self.vgs = VGS()
        self.convs = nn.ModuleList()
        for kernel in self.kernels:
            for stride in self.strides:
                if len(self.convs) < max_poolings:
                    self.convs.append(
                        nn.Conv1d(in_channels=1, out_channels=head, kernel_size=kernel, stride=stride, bias=False).to(
                            device))
                else:
                    break
            if len(self.convs) >= max_poolings:
                break

    def process_vectors(self, batch_vec1, batch_vec2):
        if batch_vec1.dim() == 2:
            batch_vec1 = batch_vec1.unsqueeze(1)
        if batch_vec2.dim() == 2:
            batch_vec2 = batch_vec2.unsqueeze(1)

        results = []
        for conv in self.convs:
            conv1 = conv(batch_vec1)
            conv2 = conv(batch_vec2)
            # print("conv1", conv1.shape)
            # print("conv2", conv2.shape)

            if conv1.nelement() != 0 and conv2.nelement() != 0:
                hamming_sim = torch.mean((torch.mean(torch.tanh(conv1) * torch.tanh(conv2), dim=-1)), dim=-1).unsqueeze(
                    -1)
                # print("hamming_sim", hamming_sim.shape)
                cosine_sim = torch.mean(self.vgs(conv1, conv2), dim=-1).unsqueeze(-1)
                # cosine_sim = torch.mean(F.cosine_similarity(conv1, conv2, dim=-1),dim=-1).unsqueeze(-1)
                results.append(torch.cat((hamming_sim, cosine_sim), dim=-1))

        return torch.cat(results, dim=-1)


import torch
import torch.nn as nn
import torch.nn.functional as F


#
# class AttentionReducer(nn.Module):
#     def __init__(self, sequence_dim):
#         super(AttentionReducer, self).__init__()
#         self.attention_weights = nn.Linear(sequence_dim, 1)
#
#     def forward(self, x):
#         # x shape: (batch, seq_len, sequence_dim)
#         weights = self.attention_weights(x).squeeze(2)  # (batch, seq_len)
#         weights = F.softmax(weights, dim=1)  # Softmax over seq_len dimension to create attention weights
#         weighted = torch.bmm(weights.unsqueeze(1), x).squeeze(1)  # (batch, sequence_dim)
#         return weighted

# class AttentionModule_w_SE(nn.Module):
#     def __init__(self, dim_size):
#         super(AttentionModule_w_SE, self).__init__()
#
#     def forward(self, x):
#         # print(x.shape,'x0')
#         mean = torch.mean(x, dim=-1)
#         transformed_global = torch.tanh(mean).unsqueeze(-1)
#         coefs = torch.sigmoid(
#             (x * transformed_global))
#         weighted= (coefs * x).sum(dim=1)
#         return weighted  # Returns the tensor in the shape (B, M) instead of weighted which is (B, L, M)

#
# class BiDAFWithAttention(nn.Module):
#     def __init__(self, embedding_dim, output_dim):
#         super(BiDAFWithAttention, self).__init__()
#         self.output_layer = nn.Linear(embedding_dim, output_dim)
#     def forward(self, passage_encodes, question_encodes):
#         sim_matrix = torch.bmm(passage_encodes, question_encodes.transpose(1, 2))
#         c2q_attn = F.softmax(sim_matrix, dim=-1)
#         context2question_attn = torch.bmm(c2q_attn, question_encodes)
#         concat_outputs = torch.tanh(passage_encodes) * torch.tanh(context2question_attn)
#         output = self.output_layer(concat_outputs).squeeze(1)
#         return output

#
# class  BiDAFWithAttention(torch.nn.Module):
#     def __init__(self, nhid,output_dim):
#         super( BiDAFWithAttention, self).__init__()
#         self.nhid = nhid
#         # self.output_layer = nn.Linear(self.nhid, output_dim)
#         self.weight = torch.nn.Parameter(torch.Tensor(self.nhid, self.nhid))
#         self.bilinear = nn.Bilinear(self.nhid, self.nhid, output_dim)
#         nn.init.xavier_uniform_(self.weight.data)
#
#     def forward(self, x, batch,mean_pool):
#         transformed_global = torch.tanh(torch.mm(mean_pool, self.weight))
#         coefs = torch.sigmoid((x * transformed_global[batch]).sum(dim=1))
#         weighted = coefs.unsqueeze(-1) * x
#         passage_encodes=global_add_pool(weighted, batch)
#         output = self.bilinear(torch.tanh(passage_encodes),torch.tanh(mean_pool))
#         return output


# class VectorSimilarity:
#     def __init__(self, max_poolings):
#         self.max_poolings = max_poolings  # 最大池化次数
#         # 根据max_poolings计算要使用的kernel和stride组合
#         self.kernels = [1, 2, 3, 4, 5, 6, 7, 8]
#         self.strides = [1, 2, 3]
#         # 为每个卷积操作创建一个单独的卷积层，总共创建max_poolings个卷积层
#         self.convs = nn.ModuleList()
#         # 填充卷积层列表，确保只创建max_poolings个卷积层
#         for kernel in self.kernels:
#             for stride in self.strides:
#                 if len(self.convs) < max_poolings:
#                     self.convs.append(
#                         nn.Conv1d(in_channels=1, out_channels=1, kernel_size=kernel, stride=stride, bias=False).to(
#                             device))
#                 else:
#                     break
#             if len(self.convs) >= max_poolings:
#                 break
#
#     def process_vectors(self, batch_vec1, batch_vec2):
#         if batch_vec1.dim() == 2:
#             batch_vec1 = batch_vec1.unsqueeze(1)  # 添加一个通道维度
#         if batch_vec2.dim() == 2:
#             batch_vec2 = batch_vec2.unsqueeze(1)  # 添加一个通道维度
#
#         results = []
#         for conv in self.convs:
#             # 应用一维卷积到两个批次
#             conv1 = conv(batch_vec1)
#             conv2 = conv(batch_vec2)
#
#             # 避免空张量
#             if conv1.nelement() != 0 and conv2.nelement() != 0:
#                 # 计算相似度
#                 hamming_sim = torch.mean(torch.tanh(conv1) * torch.tanh(conv2), dim=-1)
#                 cosine_sim = F.cosine_similarity(conv1, conv2, dim=-1)
#                 # 将结果添加到列表
#                 results.append(torch.cat((hamming_sim, cosine_sim), dim=-1))
#
#         # 将结果列表转换为张量，确保输出维度是 max_poolings 的两倍
#         return torch.cat(results, dim=-1)
#

# def pad(x, key, batch_x, batch_key):
#     key, mask_key = to_dense_batch(key, batch_key)
#     x, x_mask = to_dense_batch(x, batch_x)
#     max_seq_len = max(x.shape[1], key.shape[1])
#     # 计算两个序列的填充长度
#     padding_x = max_seq_len - x.shape[1]
#     padding_key = max_seq_len - key.shape[1]
#     # 使用torch.nn.functional.pad进行填充，更高效
#     if padding_x > 0:
#         x = F.pad(x, (0, 0, 0, padding_x))  # pad的最后两个参数是在最后一维前后填充的数量
#         # x_mask = F.pad(x_mask, (0, padding_x), value=True)  # 对mask使用True填充
#     if padding_key > 0:
#         key = F.pad(key, (0, 0, 0, padding_key))
#         # mask_key = F.pad(mask_key, (0, padding_key), value=True)
#     return x, key
#

#

class EnhancedBilinearInteraction(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels):
        """
        初始化模型。
        参数:
            in_channels (int): 输入通道数 M。
            out_channels (int): 最终输出通道数。
            hidden_channels (int): 中间层的通道数。
        """
        super(EnhancedBilinearInteraction, self).__init__()
        self.batch_norm = nn.BatchNorm1d(num_features=in_channels)

        self.bilinear = BilinearInteraction(in_channels, out_channels)

    def forward(self, x, y):
        """
        前向传播函数。
        参数:
            x (torch.Tensor): 形状为 (B, L, M) 的张量。
            y (torch.Tensor): 形状为 (B, L, M) 的张量。
        返回:
            torch.Tensor: 两个输入的双线性交互结果。
        """

        # 应用线性层
        x = self.batch_norm(x.permute(0, 2, 1)).permute(0, 2, 1)
        y = self.batch_norm(y.permute(0, 2, 1)).permute(0, 2, 1)
        # 将输入调整为卷积需要的形状 (B, C, L)
        x = x.transpose(1, 2)
        y = y.transpose(1, 2)

        # 应用双线性交互层
        output = self.bilinear(x, y)

        return output.squeeze(-1)


class BilinearInteraction(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BilinearInteraction, self).__init__()
        self.out_channels = out_channels

    def forward(self, x, y):
        B, M, L = x.shape  # y shares the same sahpe of x
        kernels = torch.tanh(y.reshape(B * M, 1, L))
        x = torch.tanh(x.reshape(1, B * M, L))
        result = F.conv1d(x, kernels, groups=B * M)
        result = result.reshape(B, M, -1)
        return result


#
# class Combinenode(nn.Module):
#     def __init__(self, dim_size):
#         super(Combinenode, self).__init__()
#         self.dim_size = dim_size
#         self.Binode = EnhancedBilinearInteraction(dim_size, dim_size, dim_size)
#
#     def forward(self, x1, batch1, x2, batch2):
#         h1, h2 = pad(x1, x2, batch1, batch2)
#         score = self.Binode(h1, h2)
#         return score


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# def mask_pad(h1, h2, batch1, batch2):
#     h1, mask1 = to_dense_batch(h1, batch1)
#     # print(mask1,"mask11")
#     h2, mask2 = to_dense_batch(h2, batch2)
#
#     max_len = max(h1.shape[1], h2.shape[1])
#
#     # 计算需要填充的长度
#     padding_h1 = max_len - h1.shape[1]
#     padding_h2 = max_len - h2.shape[1]
#
#     # 使用torch.nn.functional.pad进行填充
#     if padding_h1 > 0:
#         h1 = F.pad(h1, (0, 0, 0, padding_h1))
#         mask1 = F.pad(mask1, (0, padding_h1), value=False)  # 掩码填充时使用True
#         # print(mask1, "mask12")
#
#     if padding_h2 > 0:
#         h2 = F.pad(h2, (0, 0, 0, padding_h2))
#         mask2 = F.pad(mask2, (0, padding_h2), value=False)
#     mask1 = mask1.to(torch.bool)
#     mask2 = mask2.to(torch.bool)
#     h1 = h1.to(device)
#     h2 = h2.to(device)
#     mask1 = mask1.to(device)
#     mask2 = mask2.to(device)
#
#     return h1, h2, mask1, mask2, max_len


from typing import Optional, Tuple

import torch
from torch import Tensor

from torch_geometric.utils import scatter

from typing import Optional, Tuple

import torch
from torch import Tensor

from torch_geometric.utils import scatter


def to_dense_batch2(x: Tensor, batch: Optional[Tensor] = None,
                    fill_value: float = 0., max_num_nodes: Optional[int] = None,
                    batch_size: Optional[int] = None) -> Tuple[Tensor, Tensor, int]:
    r"""Given a sparse batch of node features
    :math:`\mathbf{X} \in \mathbb{R}^{(N_1 + \ldots + N_B) \times F}` (with
    :math:`N_i` indicating the number of nodes in graph :math:`i`), creates a
    dense node feature tensor
    :math:`\mathbf{X} \in \mathbb{R}^{B \times N_{\max} \times F}` (with
    :math:`N_{\max} = \max_i^B N_i`).
    In addition, a mask of shape :math:`\mathbf{M} \in \{ 0, 1 \}^{B \times
    N_{\max}}` is returned, holding information about the existence of
    fake-nodes in the dense representation.

    Args:
        x (Tensor): Node feature matrix
            :math:`\mathbf{X} \in \mathbb{R}^{(N_1 + \ldots + N_B) \times F}`.
        batch (LongTensor, optional): Batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            node to a specific example. Must be ordered. (default: :obj:`None`)
        fill_value (float, optional): The value for invalid entries in the
            resulting dense output tensor. (default: :obj:`0`)
        max_num_nodes (int, optional): The size of the output node dimension.
            (default: :obj:`None`)
        batch_size (int, optional) The batch size. (default: :obj:`None`)

    :rtype: (:class:`Tensor`, :class:`BoolTensor`)

    Examples:

        >>> x = torch.arange(12).view(6, 2)
        >>> x
        tensor([[ 0,  1],
                [ 2,  3],
                [ 4,  5],
                [ 6,  7],
                [ 8,  9],
                [10, 11]])

        >>> out, mask = to_dense_batch(x)
        >>> mask
        tensor([[True, True, True, True, True, True]])

        >>> batch = torch.tensor([0, 0, 1, 2, 2, 2])
        >>> out, mask = to_dense_batch(x, batch)
        >>> out
        tensor([[[ 0,  1],
                [ 2,  3],
                [ 0,  0]],
                [[ 4,  5],
                [ 0,  0],
                [ 0,  0]],
                [[ 6,  7],
                [ 8,  9],
                [10, 11]]])
        >>> mask
        tensor([[ True,  True, False],
                [ True, False, False],
                [ True,  True,  True]])

        >>> out, mask = to_dense_batch(x, batch, max_num_nodes=4)
        >>> out
        tensor([[[ 0,  1],
                [ 2,  3],
                [ 0,  0],
                [ 0,  0]],
                [[ 4,  5],
                [ 0,  0],
                [ 0,  0],
                [ 0,  0]],
                [[ 6,  7],
                [ 8,  9],
                [10, 11],
                [ 0,  0]]])

        >>> mask
        tensor([[ True,  True, False, False],
                [ True, False, False, False],
                [ True,  True,  True, False]])
    """
    if batch is None and max_num_nodes is None:
        mask = torch.ones(1, x.size(0), dtype=torch.bool, device=x.device)
        return x.unsqueeze(0), mask

    if batch is None:
        batch = x.new_zeros(x.size(0), dtype=torch.long)

    if batch_size is None:
        batch_size = int(batch.max()) + 1

    num_nodes = scatter(batch.new_ones(x.size(0)), batch, dim=0,
                        dim_size=batch_size, reduce='sum')
    cum_nodes = torch.cat([batch.new_zeros(1), num_nodes.cumsum(dim=0)])

    filter_nodes = False
    if max_num_nodes is None:
        max_num_nodes = int(num_nodes.max())
    elif num_nodes.max() > max_num_nodes:
        filter_nodes = True

    tmp = torch.arange(batch.size(0), device=x.device) - cum_nodes[batch]
    idx = tmp + (batch * max_num_nodes)
    if filter_nodes:
        mask = tmp < max_num_nodes
        x, idx = x[mask], idx[mask]

    size = [batch_size * max_num_nodes] + list(x.size())[1:]
    out = x.new_full(size, fill_value)
    out[idx] = x
    out = out.view([batch_size, max_num_nodes] + list(x.size())[1:])

    mask = torch.zeros(batch_size * max_num_nodes, dtype=torch.bool,
                       device=x.device)
    mask[idx] = 1
    mask = mask.view(batch_size, max_num_nodes)

    return out, mask, max_num_nodes


from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool


def merge_batches_and_process(x1, batch1, x2, batch2):
    """Merge two batches and process them into a dense batch with averaging."""
    # Merge x and batch
    x = torch.cat([x1, x2], dim=0)
    batch = torch.cat([batch1, batch2 + batch1.max() + 1])
    # pad = global_mean_pool(x, batch)
    batch_size = int(batch1.max()) + 1
    # 必须batch1和batch2长度相同
    assert int(batch1.max()) == int(batch2.max())

    # Create dense batch with averaging instead of zero fill
    dense_x, mask, max_num_nodes = to_dense_batch2(x, batch)
    # print("Dense features:", dense_x.shape)
    # print("Mask:", mask.shape)
    # print(batch_size, 'batch_size')

    return dense_x[0:batch_size], mask[0:batch_size], dense_x[batch_size:], mask[batch_size:], max_num_nodes


class GPSConv(torch.nn.Module):

    def __init__(
            self,
            channels: int,
            conv: Optional[MessagePassing],
            heads: int = 1,
            dropout: float = 0.0,
            act: str = 'relu',
            act_kwargs: Optional[Dict[str, Any]] = None,
            norm: Optional[str] = 'batch_norm',
            norm_kwargs: Optional[Dict[str, Any]] = None,
            attn_type: str = 'multi-head',
            attn_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()

        self.channels = channels
        self.conv = conv
        self.heads = heads
        self.dropout = dropout
        self.attn_type = attn_type

        attn_kwargs = attn_kwargs or {}
        if attn_type == 'multi-head':
            self.attn = torch.nn.MultiheadAttention(
                channels,
                heads,
                batch_first=True,
                **attn_kwargs,
            )
        elif attn_type == 'performer':  # linear attention
            self.attn = PerformerAttention(
                channels=channels,
                heads=heads,
                **attn_kwargs,
            )
        else:
            raise ValueError(f'{attn_type} is not supported')

        self.mlp = Sequential(
            Linear(channels, channels * 2),
            activation_resolver(act, **(act_kwargs or {})),
            Dropout(dropout),
            Linear(channels * 2, channels),
            Dropout(dropout),
        )

        norm_kwargs = norm_kwargs or {}
        self.norm1 = normalization_resolver(norm, channels, **norm_kwargs)
        self.norm2 = normalization_resolver(norm, channels, **norm_kwargs)
        self.norm3 = normalization_resolver(norm, channels, **norm_kwargs)

        self.norm_with_batch = False
        if self.norm1 is not None:
            signature = inspect.signature(self.norm1.forward)
            self.norm_with_batch = 'batch' in signature.parameters

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        if self.conv is not None:
            self.conv.reset_parameters()
        self.attn._reset_parameters()
        reset(self.mlp)
        if self.norm1 is not None:
            self.norm1.reset_parameters()
        if self.norm2 is not None:
            self.norm2.reset_parameters()
        if self.norm3 is not None:
            self.norm3.reset_parameters()

    def forward(self, x1, x2, edge_index1, edge_index2, batch1, batch2, **kwargs):

        hs1, hs2 = [], []
        if self.conv is not None:  # Local MPNN.
            h1 = self.conv(x1, edge_index1, **kwargs)
            h1 = h1 + x1
            if self.norm1 is not None:
                if self.norm_with_batch:
                    h1 = self.norm1(h1, batch=batch1)
                else:
                    h1 = self.norm1(h1)

            hs1.append(h1)
            h2 = self.conv(x2, edge_index2, **kwargs)
            h2 = h2 + x2
            if self.norm1 is not None:
                if self.norm_with_batch:
                    h2 = self.norm1(h2, batch=batch2)

                else:
                    h2 = self.norm1(h2)

            hs2.append(h2)
        else:
            h1, h2 = x1, x2

        h1, mask1, h2, mask2, max_len = merge_batches_and_process(h1, batch1, h2, batch2)
        # print(h1.shape, h2.shape, 'h1-h2---1')
        h_concat = torch.cat([h1, h2], dim=1)  # h  [B,max_len*2,dim]
        mask_concat = torch.cat([mask1, mask2], dim=1)
        if isinstance(self.attn, torch.nn.MultiheadAttention):
            h_concat, _ = self.attn(h_concat, h_concat, h_concat, key_padding_mask=~mask_concat,
                                    need_weights=False)
        elif isinstance(self.attn, PerformerAttention):
            h_concat = self.attn(h_concat, mask=mask_concat)
        # print(h_concat.shape,'h_concat_0')
        h1, h2 = h_concat[:, :max_len], h_concat[:, max_len:]
        # print(h1.shape,h2.shape,'h1-h2')
        h1 = h1[mask1]
        h2 = h2[mask2]
        #  the third part
        h1 = h1 + x1  # Residual connection.
        if self.norm2 is not None:
            if self.norm_with_batch:
                h1 = self.norm2(h1, batch=batch1)
            else:
                h1 = self.norm2(h1)
        hs1.append(h1)
        out1 = sum(hs1)  # Combine local and global outputs.
        out1 = out1 + self.mlp(out1)
        if self.norm3 is not None:
            if self.norm_with_batch:
                out1 = self.norm3(out1, batch=batch1)
            else:
                out1 = self.norm3(out1)
        #
        h2 = h2 + x2  # Residual connection.
        if self.norm2 is not None:
            if self.norm_with_batch:
                h2 = self.norm2(h2, batch=batch2)
            else:
                h2 = self.norm2(h2)
        hs2.append(h2)
        out2 = sum(hs2)  # Combine local and global outputs.
        out2 = out2 + self.mlp(out2)
        if self.norm3 is not None:
            if self.norm_with_batch:
                out2 = self.norm3(out2, batch=batch2)
            else:
                out2 = self.norm3(out2)

        return out1, out2
