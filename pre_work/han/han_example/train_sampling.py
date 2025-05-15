# -*- coding: utf-8 -*-
"""
HAN 小批量训练，使用 RandomWalkSampler 进行采样。
注意：此示例使用 RandomWalkSampler 对邻居节点进行采样，在验证或测试时很难获取所有邻居节点，
因此在验证/测试时采样的邻居节点数量是训练时的两倍。
"""
# 导入 argparse 模块，用于解析命令行参数
import argparse
# 导入 DGL 库，用于图神经网络任务
import dgl
# 导入 numpy 库，用于数值计算
import numpy
# 导入 PyTorch 库，用于深度学习任务
import torch
# 从 PyTorch 中导入神经网络模块
import torch.nn as nn
# 从 PyTorch 中导入神经网络函数模块
import torch.nn.functional as F
# 从 DGL 的 PyTorch 模块中导入 GAT 卷积层
from dgl.nn.pytorch import GATConv
# 从 DGL 的采样模块中导入随机游走邻居采样器
from dgl.sampling import RandomWalkNeighborSampler
# 从 model_hetero 模块中导入语义注意力模块
from model_hetero import SemanticAttention
# 从 sklearn.metrics 模块中导入 f1_score 函数，用于计算 F1 分数
from sklearn.metrics import f1_score
# 从 torch.utils.DBLP 模块中导入 DataLoader 类，用于批量加载数据
from torch.utils.data import DataLoader
# 从 utils 模块中导入早停机制类和设置随机种子的函数
from utils import EarlyStopping, set_random_seed


# 定义 HAN 层模块
class HANLayer(torch.nn.Module):
    """
    HAN 层。

    参数
    ---------
    num_metapath : 基于元路径的子图数量
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

    def __init__(
        self, num_metapath, in_size, out_size, layer_num_heads, dropout
    ):
        # 调用父类的构造函数
        super(HANLayer, self).__init__()

        # 为每个元路径对应的邻接矩阵定义一个 GAT 层
        self.gat_layers = nn.ModuleList()
        for i in range(num_metapath):
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
        # 保存元路径的数量
        self.num_metapath = num_metapath

    def forward(self, block_list, h_list):
        # 存储不同元路径下的语义嵌入
        semantic_embeddings = []

        # 遍历每个块
        for i, block in enumerate(block_list):
            # 通过 GAT 层计算语义嵌入，并将多个注意力头的输出展平
            semantic_embeddings.append(
                self.gat_layers[i](block, h_list[i]).flatten(1)
            )
        # 将不同元路径下的语义嵌入堆叠在一起
        semantic_embeddings = torch.stack(
            semantic_embeddings, dim=1
        )  # (N, M, D * K)

        # 通过语义注意力模块聚合不同元路径下的语义嵌入
        return self.semantic_attention(semantic_embeddings)  # (N, D * K)


# 定义 HAN 模型
class HAN(nn.Module):
    def __init__(
        self, num_metapath, in_size, hidden_size, out_size, num_heads, dropout
    ):
        # 调用父类的构造函数
        super(HAN, self).__init__()

        # 定义神经网络层列表
        self.layers = nn.ModuleList()
        # 添加第一个 HAN 层
        self.layers.append(
            HANLayer(num_metapath, in_size, hidden_size, num_heads[0], dropout)
        )
        # 循环添加后续的 HAN 层
        for l in range(1, len(num_heads)):
            self.layers.append(
                HANLayer(
                    # 元路径的数量
                    num_metapath,
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


# 定义 HAN 采样器类
class HANSampler(object):
    def __init__(self, g, metapath_list, num_neighbors):
        # 初始化采样器列表
        self.sampler_list = []
        # 为每个元路径创建一个随机游走邻居采样器
        for metapath in metapath_list:
            # 注意：随机游走可能会得到相同的路径（相同的边），这些边会在采样图中被移除。
            # 因此，采样图中的边数可能少于 num_random_walks（num_neighbors）。
            self.sampler_list.append(
                RandomWalkNeighborSampler(
                    # 输入图
                    G=g,
                    # 随机游走的遍历次数
                    num_traversals=1,
                    # 随机游走的终止概率
                    termination_prob=0,
                    # 随机游走的次数
                    num_random_walks=num_neighbors,
                    # 采样的邻居节点数量
                    num_neighbors=num_neighbors,
                    # 元路径
                    metapath=metapath,
                )
            )

    def sample_blocks(self, seeds):
        # 初始化块列表
        block_list = []
        # 遍历每个采样器
        for sampler in self.sampler_list:
            # 采样得到前沿图
            frontier = sampler(seeds)
            # 移除自环
            frontier = dgl.remove_self_loop(frontier)
            # 添加自环
            frontier.add_edges(torch.tensor(seeds), torch.tensor(seeds))
            # 将前沿图转换为块
            block = dgl.to_block(frontier, seeds)
            block_list.append(block)

        return seeds, block_list


# 定义一个函数，用于计算准确率、微观 F1 分数和宏观 F1 分数
def score(logits, labels):
    # 在 logits 张量的第一个维度上找到最大值的索引
    _, indices = torch.max(logits, dim=1)
    # 将索引转换为长整型并移动到 CPU 上，然后转换为 NumPy 数组
    prediction = indices.long().cpu().numpy()
    # 将标签移动到 CPU 上，然后转换为 NumPy 数组
    labels = labels.cpu().numpy()

    # 计算预测正确的样本数占总样本数的比例，即准确率
    accuracy = (prediction == labels).sum() / len(prediction)
    # 计算微观 F1 分数
    micro_f1 = f1_score(labels, prediction, average="micro")
    # 计算宏观 F1 分数
    macro_f1 = f1_score(labels, prediction, average="macro")

    # 返回准确率、微观 F1 分数和宏观 F1 分数
    return accuracy, micro_f1, macro_f1


# 定义一个函数，用于评估模型在验证集或测试集上的性能
def evaluate(
    model,
    g,
    metapath_list,
    num_neighbors,
    features,
    labels,
    val_nid,
    loss_fcn,
    batch_size,
):
    # 将模型设置为评估模式，关闭 Dropout 等训练时使用的特殊层
    model.eval()

    # 创建验证集的 HAN 采样器，采样的邻居节点数量是训练时的两倍
    han_valid_sampler = HANSampler(
        g, metapath_list, num_neighbors=num_neighbors * 2
    )
    # 创建数据加载器，用于批量加载验证集数据
    dataloader = DataLoader(
        # 验证集节点索引
        dataset=val_nid,
        # 批量大小
        batch_size=batch_size,
        # 自定义数据处理函数
        collate_fn=han_valid_sampler.sample_blocks,
        # 不打乱数据顺序
        shuffle=False,
        # 最后一个批次数据不足时不丢弃
        drop_last=False,
        # 数据加载的线程数
        num_workers=4,
    )
    # 初始化正确预测的样本数和总样本数
    correct = total = 0
    # 初始化预测结果列表和标签列表
    prediction_list = []
    labels_list = []
    # 上下文管理器，不计算梯度，提高计算效率
    with torch.no_grad():
        # 遍历数据加载器中的每个批次
        for step, (seeds, blocks) in enumerate(dataloader):
            # 加载子图的特征
            h_list = load_subtensors(blocks, features)
            # 将块移动到指定设备上
            blocks = [block.to(args["device"]) for block in blocks]
            # 将特征移动到指定设备上
            hs = [h.to(args["device"]) for h in h_list]

            # 前向传播，得到模型的输出
            logits = model(blocks, hs)
            # 计算损失值
            loss = loss_fcn(
                logits, labels[numpy.asarray(seeds)].to(args["device"])
            )
            # 获取每个样本的预测标签
            _, indices = torch.max(logits, dim=1)
            prediction = indices.long().cpu().numpy()
            labels_batch = labels[numpy.asarray(seeds)].cpu().numpy()

            # 将预测结果和标签添加到列表中
            prediction_list.append(prediction)
            labels_list.append(labels_batch)

            # 累加正确预测的样本数和总样本数
            correct += (prediction == labels_batch).sum()
            total += prediction.shape[0]

    # 将所有批次的预测结果和标签拼接在一起
    total_prediction = numpy.concatenate(prediction_list)
    total_labels = numpy.concatenate(labels_list)
    # 计算微观 F1 分数
    micro_f1 = f1_score(total_labels, total_prediction, average="micro")
    # 计算宏观 F1 分数
    macro_f1 = f1_score(total_labels, total_prediction, average="macro")
    # 计算准确率
    accuracy = correct / total

    return loss, accuracy, micro_f1, macro_f1


# 定义一个函数，用于加载子图的特征
def load_subtensors(blocks, features):
    # 初始化特征列表
    h_list = []
    # 遍历每个块
    for block in blocks:
        # 获取块的源节点索引
        input_nodes = block.srcdata[dgl.NID]
        # 根据源节点索引获取对应的特征
        h_list.append(features[input_nodes])
    return h_list


# 定义主函数，包含模型训练和评估的主要逻辑
def main(args):
    # 如果使用 ACM 数据集
    if args["dataset"] == "ACMRaw":
        # 从 utils 模块中导入加载数据的函数
        from utils import load_data

        # 加载数据集并返回相关数据
        (
            g,
            features,
            labels,
            n_classes,
            train_nid,
            val_nid,
            test_nid,
            train_mask,
            val_mask,
            test_mask,
        ) = load_data("ACMRaw")
        # 定义元路径列表
        metapath_list = [["pa", "ap"], ["pf", "fp"]]
    else:
        # 如果使用不支持的数据集，抛出异常
        raise NotImplementedError(
            "Unsupported dataset {}".format(args["dataset"])
        )

    # 是否需要为不同的基于元路径的图设置不同的邻居节点数量？
    num_neighbors = args["num_neighbors"]
    # 创建训练集的 HAN 采样器
    han_sampler = HANSampler(g, metapath_list, num_neighbors)
    # 创建 PyTorch 数据加载器，用于构建块
    dataloader = DataLoader(
        # 训练集节点索引
        dataset=train_nid,
        # 批量大小
        batch_size=args["batch_size"],
        # 自定义数据处理函数
        collate_fn=han_sampler.sample_blocks,
        # 打乱数据顺序
        shuffle=True,
        # 最后一个批次数据不足时不丢弃
        drop_last=False,
        # 数据加载的线程数
        num_workers=4,
    )

    # 初始化 HAN 模型，并将其移动到指定设备上
    model = HAN(
        num_metapath=len(metapath_list),
        in_size=features.shape[1],
        hidden_size=args["hidden_units"],
        out_size=n_classes,
        num_heads=args["num_heads"],
        dropout=args["dropout"],
    ).to(args["device"])

    # 计算模型的总参数数量
    total_params = sum(p.numel() for p in model.parameters())
    print("total_params: {:d}".format(total_params))
    # 计算模型的可训练参数数量
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print("total trainable params: {:d}".format(total_trainable_params))

    # 初始化早停机制，设置耐心值
    stopper = EarlyStopping(patience=args["patience"])
    # 定义损失函数，使用交叉熵损失
    loss_fn = torch.nn.CrossEntropyLoss()
    # 定义优化器，使用 Adam 优化器
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args["lr"], weight_decay=args["weight_decay"]
    )

    # 开始训练循环，迭代指定的轮数
    for epoch in range(args["num_epochs"]):
        # 将模型设置为训练模式，开启 Dropout 等训练时使用的特殊层
        model.train()
        # 遍历数据加载器中的每个批次
        for step, (seeds, blocks) in enumerate(dataloader):
            # 加载子图的特征
            h_list = load_subtensors(blocks, features)
            # 将块移动到指定设备上
            blocks = [block.to(args["device"]) for block in blocks]
            # 将特征移动到指定设备上
            hs = [h.to(args["device"]) for h in h_list]

            # 前向传播，得到模型的输出
            logits = model(blocks, hs)
            # 计算训练损失
            loss = loss_fn(
                logits, labels[numpy.asarray(seeds)].to(args["device"])
            )

            # 梯度清零，避免梯度累积
            optimizer.zero_grad()
            # 反向传播，计算梯度
            loss.backward()
            # 更新模型参数
            optimizer.step()

            # 打印每个批次的训练信息
            train_acc, train_micro_f1, train_macro_f1 = score(
                logits, labels[numpy.asarray(seeds)]
            )
            print(
                "Epoch {:d} | loss: {:.4f} | train_acc: {:.4f} | train_micro_f1: {:.4f} | train_macro_f1: {:.4f}".format(
                    epoch + 1, loss, train_acc, train_micro_f1, train_macro_f1
                )
            )
        # 调用 evaluate 函数，计算验证集的损失值、准确率、微观 F1 分数和宏观 F1 分数
        val_loss, val_acc, val_micro_f1, val_macro_f1 = evaluate(
            model,
            g,
            metapath_list,
            num_neighbors,
            features,
            labels,
            val_nid,
            loss_fn,
            args["batch_size"],
        )
        # 调用早停机制的 step 方法，判断是否需要提前停止训练
        early_stop = stopper.step(val_loss.data.item(), val_acc, model)

        # 打印验证集的指标
        print(
            "Epoch {:d} | Val loss {:.4f} | Val Accuracy {:.4f} | Val Micro f1 {:.4f} | Val Macro f1 {:.4f}".format(
                epoch + 1, val_loss.item(), val_acc, val_micro_f1, val_macro_f1
            )
        )

        # 如果满足早停条件，跳出训练循环
        if early_stop:
            break

    # 加载早停机制保存的最佳模型参数
    stopper.load_checkpoint(model)
    # 调用 evaluate 函数，计算测试集的损失值、准确率、微观 F1 分数和宏观 F1 分数
    test_loss, test_acc, test_micro_f1, test_macro_f1 = evaluate(
        model,
        g,
        metapath_list,
        num_neighbors,
        features,
        labels,
        test_nid,
        loss_fn,
        args["batch_size"],
    )
    # 打印测试集的指标
    print(
        "Test loss {:.4f} | Test Accuracy {:.4f} | Test Micro f1 {:.4f} | Test Macro f1 {:.4f}".format(
            test_loss.item(), test_acc, test_micro_f1, test_macro_f1
        )
    )


if __name__ == "__main__":
    # 创建一个 ArgumentParser 对象，用于解析命令行参数
    parser = argparse.ArgumentParser("mini-batch HAN")
    # 添加一个名为 -s 或 --seed 的参数，用于设置随机种子，默认值为 1
    parser.add_argument("-s", "--seed", type=int, default=1, help="Random seed")
    # 添加一个名为 --batch_size 的参数，用于设置批量大小，默认值为 32
    parser.add_argument("--batch_size", type=int, default=32)
    # 添加一个名为 --num_neighbors 的参数，用于设置采样的邻居节点数量，默认值为 20
    parser.add_argument("--num_neighbors", type=int, default=20)
    # 添加一个名为 --lr 的参数，用于设置学习率，默认值为 0.001
    parser.add_argument("--lr", type=float, default=0.001)
    # 添加一个名为 --num_heads 的参数，用于设置注意力头的数量，默认值为 [8]
    parser.add_argument("--num_heads", type=list, default=[8])
    # 添加一个名为 --hidden_units 的参数，用于设置隐藏层的单元数量，默认值为 8
    parser.add_argument("--hidden_units", type=int, default=8)
    # 添加一个名为 --dropout 的参数，用于设置 Dropout 概率，默认值为 0.6
    parser.add_argument("--dropout", type=float, default=0.6)
    # 添加一个名为 --weight_decay 的参数，用于设置权重衰减，默认值为 0.001
    parser.add_argument("--weight_decay", type=float, default=0.001)
    # 添加一个名为 --num_epochs 的参数，用于设置训练的轮数，默认值为 100
    parser.add_argument("--num_epochs", type=int, default=100)
    # 添加一个名为 --patience 的参数，用于设置早停机制的耐心值，默认值为 10
    parser.add_argument("--patience", type=int, default=10)
    # 添加一个名为 --dataset 的参数，用于设置数据集名称，默认值为 "ACMRaw"
    parser.add_argument("--dataset", type=str, default="ACMRaw")
    # 添加一个名为 --device 的参数，用于设置训练设备，默认值为 "cuda:0"
    parser.add_argument("--device", type=str, default="cuda:0")

    # 解析命令行参数，并将结果转换为字典
    args = parser.parse_args().__dict__
    # 调用设置随机种子的函数
    # set_random_seed(args['seed'])

    # 调用 main 函数，开始训练和评估模型
    main(args)
