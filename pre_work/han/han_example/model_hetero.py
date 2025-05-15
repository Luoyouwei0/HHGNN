"""
此模型展示了在原始异构图上使用 dgl.metapath_reachable_graph 的示例。

由于原始 HAN 实现仅提供了预处理后的同构图，此模型无法复现 HAN 论文中的结果，
因为他们没有提供预处理代码，并且我们从 ACM 构建了另一个数据集，
包含不同的论文集合、连接关系、特征和标签。
"""

# 导入 DGL 库，用于图神经网络任务
import dgl
# 导入 PyTorch 库，用于深度学习任务
import torch
# 从 PyTorch 中导入神经网络模块
import torch.nn as nn
# 从 PyTorch 中导入神经网络函数模块
import torch.nn.functional as F
# 从 DGL 的 PyTorch 模块中导入 GAT 卷积层
from dgl.nn.pytorch import GATConv


# 定义语义注意力模块，用于在不同元路径上聚合特征
class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        # 调用父类的构造函数
        super(SemanticAttention, self).__init__()

        # 定义一个顺序模块，包含线性层、Tanh 激活函数和另一个线性层
        self.project = nn.Sequential(
            # 输入维度为 in_size，输出维度为 hidden_size 的线性层
            nn.Linear(in_size, hidden_size),
            # Tanh 激活函数
            nn.Tanh(),
            # 输入维度为 hidden_size，输出维度为 1 的线性层，不使用偏置
            nn.Linear(hidden_size, 1, bias=False),
        )

    def forward(self, z):
        # 通过 project 模块计算注意力权重，并在第 0 维求平均
        w = self.project(z).mean(0)  # (M, 1)
        # 对注意力权重进行 softmax 操作，得到归一化的注意力系数
        beta = torch.softmax(w, dim=0)  # (M, 1)
        # 扩展注意力系数的维度，使其与输入特征 z 的维度匹配
        beta = beta.expand((z.shape[0],) + beta.shape)  # (N, M, 1)

        # 根据注意力系数对输入特征进行加权求和
        return (beta * z).sum(1)  # (N, D * K)


# 定义 HAN 层模块
class HANLayer(nn.Module):
    """
    HAN 层。

    参数
    ---------
    meta_paths : 元路径列表，每个元路径是一个边类型列表
    in_size : 输入特征维度
    out_size : 输出特征维度
    layer_num_heads : 注意力头的数量
    dropout : Dropout 概率

    输入
    ------
    g : DGLGraph
        异构图
    h : 张量
        输入特征

    输出
    -------
    张量
        输出特征
    """

    def __init__(self, meta_paths, in_size, out_size, layer_num_heads, dropout):
        # 调用父类的构造函数
        super(HANLayer, self).__init__()

        # 为每个元路径对应的邻接矩阵定义一个 GAT 层
        self.gat_layers = nn.ModuleList()
        for i in range(len(meta_paths)):
            self.gat_layers.append(
                GATConv(
                    # 输入特征维度
                    in_size,
                    # 输出特征维度
                    out_size,
                    # 注意力头的数量
                    layer_num_heads,
                    # 输入特征的 Dropout 概率
                    dropout,
                    # 注意力权重的 Dropout 概率
                    dropout,
                    # 激活函数，使用 ELU
                    activation=F.elu,
                    # 允许入度为 0 的节点
                    allow_zero_in_degree=True,
                )
            )
        # 定义语义注意力模块，输入维度为 out_size * layer_num_heads
        self.semantic_attention = SemanticAttention(
            in_size=out_size * layer_num_heads
        )
        # 将元路径列表转换为元组列表，方便后续使用
        self.meta_paths = list(tuple(meta_path) for meta_path in meta_paths)

        # 缓存图和合并后的图
        self._cached_graph = None
        self._cached_coalesced_graph = {}

    def forward(self, g, h):
        # 存储不同元路径下的语义嵌入
        semantic_embeddings = []

        # 如果缓存的图为空或者与当前图不一致，则更新缓存
        if self._cached_graph is None or self._cached_graph is not g:
            self._cached_graph = g
            self._cached_coalesced_graph.clear()
            for meta_path in self.meta_paths:
                # 计算每个元路径可达的图，并缓存结果
                self._cached_coalesced_graph[
                    meta_path
                ] = dgl.metapath_reachable_graph(g, meta_path)

        # 遍历每个元路径
        for i, meta_path in enumerate(self.meta_paths):
            # 获取当前元路径对应的合并后的图
            new_g = self._cached_coalesced_graph[meta_path]
            # 通过 GAT 层计算语义嵌入，并将多个注意力头的输出展平
            semantic_embeddings.append(self.gat_layers[i](new_g, h).flatten(1))
        # 将不同元路径下的语义嵌入堆叠在一起
        semantic_embeddings = torch.stack(
            semantic_embeddings, dim=1
        )  # (N, M, D * K)

        # 通过语义注意力模块聚合不同元路径下的语义嵌入
        return self.semantic_attention(semantic_embeddings)  # (N, D * K)


# 定义 HAN 模型
class HAN(nn.Module):
    def __init__(
        self, meta_paths, in_size, hidden_size, out_size, num_heads, dropout
    ):
        # 调用父类的构造函数
        super(HAN, self).__init__()

        # 定义神经网络层列表
        self.layers = nn.ModuleList()
        # 添加第一个 HAN 层
        self.layers.append(
            HANLayer(meta_paths, in_size, hidden_size, num_heads[0], dropout)
        )
        # 循环添加后续的 HAN 层
        for l in range(1, len(num_heads)):
            self.layers.append(
                HANLayer(
                    # 元路径列表
                    meta_paths,
                    # 输入特征维度，为上一层输出维度乘以注意力头的数量
                    hidden_size * num_heads[l - 1],
                    # 输出特征维度
                    hidden_size,
                    # 注意力头的数量
                    num_heads[l],
                    # Dropout 概率
                    dropout,
                )
            )
        # 定义最终的预测层，将特征映射到输出类别维度
        self.predict = nn.Linear(hidden_size * num_heads[-1], out_size)

    def forward(self, g, h):
        # 依次通过每一层 HAN 层
        for gnn in self.layers:
            h = gnn(g, h)

        # 通过预测层得到最终的输出
        return self.predict(h)
