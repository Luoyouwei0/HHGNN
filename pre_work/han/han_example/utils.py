# 导入 datetime 模块，用于处理日期和时间
import datetime
# 导入 errno 模块，用于处理系统错误码
import errno
# 导入 os 模块，用于与操作系统进行交互
import os
# 导入 pickle 模块，用于对象的序列化和反序列化
import pickle
# 导入 random 模块，用于生成随机数
import random
# 从 pprint 模块中导入 pprint 函数，用于美观地打印数据结构
from pprint import pprint

# 导入 DGL 库，用于图神经网络任务
import dgl

# 导入 numpy 库，用于数值计算
import numpy as np
# 导入 PyTorch 库，用于深度学习任务
import torch
# 从 DGL 的数据工具模块中导入相关函数，用于下载和获取数据目录
from dgl.data.utils import _get_dgl_url, download, get_download_dir
# 从 scipy 库中导入 io 模块，用于处理各种文件格式
from scipy import io as sio, sparse


# 定义一个函数，用于设置随机种子，确保实验的可重复性
def set_random_seed(seed=0):
    """
    设置随机种子。

    参数
    ----------
    seed : int
        要使用的随机种子
    """
    # 设置 Python 内置的随机数生成器的种子
    random.seed(seed)
    # 设置 NumPy 的随机数生成器的种子
    np.random.seed(seed)
    # 设置 PyTorch 的 CPU 随机数生成器的种子
    torch.manual_seed(seed)
    # 如果有可用的 GPU，设置 PyTorch 的 GPU 随机数生成器的种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


# 定义一个函数，用于创建指定路径的目录
def mkdir_p(path, log=True):
    """
    为指定路径创建一个目录。

    参数
    ----------
    path : str
        目录路径名称
    log : bool
        是否打印目录创建结果
    """
    try:
        # 递归创建目录
        os.makedirs(path)
        if log:
            print("Created directory {}".format(path))
    except OSError as exc:
        # 如果目录已存在，且路径是一个目录，则打印提示信息
        if exc.errno == errno.EEXIST and os.path.isdir(path) and log:
            print("Directory {} already exists.".format(path))
        else:
            # 其他错误则抛出异常
            raise


# 定义一个函数，用于获取基于日期的目录名后缀
def get_date_postfix():
    """
    获取基于日期的目录名后缀。

    返回
    -------
    post_fix : str
        日期后缀字符串
    """
    # 获取当前日期和时间
    dt = datetime.datetime.now()
    # 格式化日期和时间，生成后缀字符串
    post_fix = "{}_{:02d}-{:02d}-{:02d}".format(
        dt.date(), dt.hour, dt.minute, dt.second
    )

    return post_fix


# 定义一个函数，用于命名和创建日志目录
def setup_log_dir(args, sampling=False):
    """
    命名并创建用于日志记录的目录。

    参数
    ----------
    args : dict
        配置参数
    sampling : bool
        是否使用基于采样的训练

    返回
    -------
    log_dir : str
        日志目录的路径
    """
    # 获取日期后缀
    date_postfix = get_date_postfix()
    # 拼接日志目录路径
    log_dir = os.path.join(
        args["log_dir"], "{}_{}".format(args["dataset"], date_postfix)
    )

    if sampling:
        # 如果使用采样训练，在目录名后添加 "_sampling"
        log_dir = log_dir + "_sampling"

    # 创建日志目录
    mkdir_p(log_dir)
    return log_dir


# 论文中的默认配置
default_configure = {
    "lr": 0.005,  # 学习率
    "num_heads": [8],  # 节点级注意力的注意力头数量
    "hidden_units": 8,  # 隐藏单元数量
    "dropout": 0.6,  # Dropout 概率
    "weight_decay": 0.001,  # 权重衰减
    "num_epochs": 200,  # 训练轮数
    "patience": 100,  # 早停机制的耐心值
}

# 采样训练的配置
sampling_configure = {"batch_size": 20}


# 定义一个函数，用于设置实验配置
def setup(args):
    """
    设置实验配置。

    参数
    ----------
    args : dict
        原始配置参数

    返回
    -------
    args : dict
        更新后的配置参数
    """
    # 将默认配置更新到参数中
    args.update(default_configure)
    # 设置随机种子
    set_random_seed(args["seed"])
    # 根据是否使用异构图设置数据集名称
    args["dataset"] = "ACMRaw" if args["hetero"] else "ACM"
    # 根据是否有可用的 GPU 设置训练设备
    args["device"] = "cuda:0" if torch.cuda.is_available() else "cpu"
    # 创建日志目录并更新到参数中
    args["log_dir"] = setup_log_dir(args)
    return args


# 定义一个函数，用于设置采样训练的实验配置
def setup_for_sampling(args):
    """
    设置采样训练的实验配置。

    参数
    ----------
    args : dict
        原始配置参数

    返回
    -------
    args : dict
        更新后的配置参数
    """
    # 将默认配置更新到参数中
    args.update(default_configure)
    # 将采样训练的配置更新到参数中
    args.update(sampling_configure)
    # 设置随机种子
    set_random_seed()
    # 根据是否有可用的 GPU 设置训练设备
    args["device"] = "cuda:0" if torch.cuda.is_available() else "cpu"
    # 创建采样训练的日志目录并更新到参数中
    args["log_dir"] = setup_log_dir(args, sampling=True)
    return args


# 定义一个函数，用于生成二进制掩码
def get_binary_mask(total_size, indices):
    """
    生成二进制掩码。

    参数
    ----------
    total_size : int
        掩码的总长度
    indices : list 或 tensor
        需要设置为 1 的索引

    返回
    -------
    mask : torch.Tensor
        二进制掩码张量
    """
    # 初始化全 0 张量
    mask = torch.zeros(total_size)
    # 将指定索引位置的值设置为 1
    mask[indices] = 1
    return mask.byte()


# 定义一个函数，用于加载 ACM 数据集
def load_acm(remove_self_loop):
    """
    加载 ACM 数据集。

    参数
    ----------
    remove_self_loop : bool
        是否移除自环

    返回
    -------
    gs : list
        元路径邻居的邻接矩阵图列表
    features : torch.Tensor
        特征张量
    labels : torch.Tensor
        标签张量
    num_classes : int
        类别数量
    train_idx : torch.Tensor
        训练集索引
    val_idx : torch.Tensor
        验证集索引
    test_idx : torch.Tensor
        测试集索引
    train_mask : torch.Tensor
        训练集掩码
    val_mask : torch.Tensor
        验证集掩码
    test_mask : torch.Tensor
        测试集掩码
    """
    # 数据集下载链接
    url = "dataset/ACM3025.pkl"
    # 数据集保存路径
    data_path = get_download_dir() + "/ACM3025.pkl"
    # 下载数据集
    download(_get_dgl_url(url), path=data_path)

    with open(data_path, "rb") as f:
        # 加载数据集
        data = pickle.load(f)

    # 提取标签和特征，并转换为 PyTorch 张量
    labels, features = (
        torch.from_numpy(data["label"].todense()).long(),
        torch.from_numpy(data["feature"].todense()).float(),
    )
    # 获取类别数量
    num_classes = labels.shape[1]
    # 将 one-hot 编码的标签转换为类别索引
    labels = labels.nonzero()[:, 1]

    if remove_self_loop:
        # 如果需要移除自环，从邻接矩阵中减去单位矩阵
        num_nodes = data["label"].shape[0]
        data["PAP"] = sparse.csr_matrix(data["PAP"] - np.eye(num_nodes))
        data["PLP"] = sparse.csr_matrix(data["PLP"] - np.eye(num_nodes))

    # 基于元路径的邻居的邻接矩阵图
    author_g = dgl.from_scipy(data["PAP"])
    subject_g = dgl.from_scipy(data["PLP"])
    gs = [author_g, subject_g]

    # 提取训练集、验证集和测试集的索引
    train_idx = torch.from_numpy(data["train_idx"]).long().squeeze(0)
    val_idx = torch.from_numpy(data["val_idx"]).long().squeeze(0)
    test_idx = torch.from_numpy(data["test_idx"]).long().squeeze(0)

    # 获取节点数量
    num_nodes = author_g.num_nodes()
    # 生成训练集、验证集和测试集的掩码
    train_mask = get_binary_mask(num_nodes, train_idx)
    val_mask = get_binary_mask(num_nodes, val_idx)
    test_mask = get_binary_mask(num_nodes, test_idx)

    print("dataset loaded")
    # 打印数据集划分比例
    pprint(
        {
            "dataset": "ACM",
            "train": train_mask.sum().item() / num_nodes,
            "val": val_mask.sum().item() / num_nodes,
            "test": test_mask.sum().item() / num_nodes,
        }
    )

    return (
        gs,
        features,
        labels,
        num_classes,
        train_idx,
        val_idx,
        test_idx,
        train_mask,
        val_mask,
        test_mask,
    )


# 定义一个函数，用于加载原始的 ACM 数据集
def load_acm_raw(remove_self_loop):
    """
    加载原始的 ACM 数据集。

    参数
    ----------
    remove_self_loop : bool
        是否移除自环（当前要求不能移除）

    返回
    -------
    hg : dgl.DGLHeteroGraph
        异构图
    features : torch.Tensor
        特征张量
    labels : torch.Tensor
        标签张量
    num_classes : int
        类别数量
    train_idx : torch.Tensor
        训练集索引
    val_idx : torch.Tensor
        验证集索引
    test_idx : torch.Tensor
        测试集索引
    train_mask : torch.Tensor
        训练集掩码
    val_mask : torch.Tensor
        验证集掩码
    test_mask : torch.Tensor
        测试集掩码
    """
    assert not remove_self_loop
    # 数据集下载链接
    url = "dataset/ACM.mat"
    # 数据集保存路径
    data_path = get_download_dir() + "/ACM.mat"
    # 下载数据集
    download(_get_dgl_url(url), path=data_path)

    # 加载 .mat 格式的数据集
    data = sio.loadmat(data_path)
    p_vs_l = data["PvsL"]  # 论文 - 领域关系矩阵
    p_vs_a = data["PvsA"]  # 论文 - 作者关系矩阵
    p_vs_t = data["PvsT"]  # 论文 - 术语关系矩阵，词袋模型
    p_vs_c = data["PvsC"]  # 论文 - 会议关系矩阵，标签来源于此

    # We assign
    # (1) KDD papers as class 0 (DBLP mining),
    # (2) SIGMOD and VLDB papers as class 1 (database),
    # (3) SIGCOMM and MOBICOMM papers as class 2 (communication)
    conf_ids = [0, 1, 9, 10, 13]
    label_ids = [0, 1, 2, 2, 1]

    # 筛选出指定会议的论文
    p_vs_c_filter = p_vs_c[:, conf_ids]
    p_selected = (p_vs_c_filter.sum(1) != 0).A1.nonzero()[0]
    p_vs_l = p_vs_l[p_selected]
    p_vs_a = p_vs_a[p_selected]
    p_vs_t = p_vs_t[p_selected]
    p_vs_c = p_vs_c[p_selected]

    # 创建异构图
    hg = dgl.heterograph(
        {
            ("paper", "pa", "author"): p_vs_a.nonzero(),
            ("author", "ap", "paper"): p_vs_a.transpose().nonzero(),
            ("paper", "pf", "field"): p_vs_l.nonzero(),
            ("field", "fp", "paper"): p_vs_l.transpose().nonzero(),
        }
    )

    # 将论文 - 术语关系矩阵转换为特征张量
    features = torch.FloatTensor(p_vs_t.toarray())

    # 提取论文 - 会议关系矩阵的非零元素的行和列索引
    pc_p, pc_c = p_vs_c.nonzero()
    # 初始化标签数组
    labels = np.zeros(len(p_selected), dtype=np.int64)
    # 根据会议 ID 分配类别标签
    for conf_id, label_id in zip(conf_ids, label_ids):
        labels[pc_p[pc_c == conf_id]] = label_id
    # 将标签数组转换为 PyTorch 张量
    labels = torch.LongTensor(labels)

    # 类别数量
    num_classes = 3

    # 初始化浮点数掩码数组
    float_mask = np.zeros(len(pc_p))
    # 为每个会议的论文随机分配 0 到 1 之间的值
    for conf_id in conf_ids:
        pc_c_mask = pc_c == conf_id
        float_mask[pc_c_mask] = np.random.permutation(
            np.linspace(0, 1, pc_c_mask.sum())
        )
    # 根据浮点数掩码划分训练集、验证集和测试集
    train_idx = np.where(float_mask <= 0.2)[0]
    val_idx = np.where((float_mask > 0.2) & (float_mask <= 0.3))[0]
    test_idx = np.where(float_mask > 0.3)[0]

    # 获取论文节点数量
    num_nodes = hg.num_nodes("paper")
    # 生成训练集、验证集和测试集的掩码
    train_mask = get_binary_mask(num_nodes, train_idx)
    val_mask = get_binary_mask(num_nodes, val_idx)
    test_mask = get_binary_mask(num_nodes, test_idx)

    return (
        hg,
        features,
        labels,
        num_classes,
        train_idx,
        val_idx,
        test_idx,
        train_mask,
        val_mask,
        test_mask,
    )


# 定义一个函数，用于加载指定数据集
def load_data(dataset, remove_self_loop=False):
    """
    加载指定数据集。

    参数
    ----------
    dataset : str
        数据集名称，支持 "ACM" 和 "ACMRaw"
    remove_self_loop : bool
        是否移除自环

    返回
    -------
    数据集相关信息，具体返回值取决于数据集类型
    """
    if dataset == "ACM":
        return load_acm(remove_self_loop)
    elif dataset == "ACMRaw":
        return load_acm_raw(remove_self_loop)
    else:
        return NotImplementedError("Unsupported dataset {}".format(dataset))


# 定义一个早停机制类，用于在验证集性能不再提升时提前停止训练
class EarlyStopping(object):
    def __init__(self, patience=10):
        """
        初始化早停机制。

        参数
        ----------
        patience : int
            耐心值，即验证集性能不再提升时继续训练的轮数
        """
        # 获取当前日期和时间
        dt = datetime.datetime.now()
        # 生成保存模型的文件名
        self.filename = "early_stop_{}_{:02d}-{:02d}-{:02d}.pth".format(
            dt.date(), dt.hour, dt.minute, dt.second
        )
        # 耐心值
        self.patience = patience
        # 计数器，记录验证集性能未提升的轮数
        self.counter = 0
        # 最佳准确率
        self.best_acc = None
        # 最佳损失值
        self.best_loss = None
        # 是否提前停止训练的标志
        self.early_stop = False

    def step(self, loss, acc, model):
        """
        执行早停机制的一步，根据当前的损失值和准确率判断是否需要提前停止训练。

        参数
        ----------
        loss : float
            当前验证集的损失值
        acc : float
            当前验证集的准确率
        model : torch.nn.Module
            模型

        返回
        -------
        early_stop : bool
            是否需要提前停止训练
        """
        if self.best_loss is None:
            # 第一次执行时，保存当前的准确率和损失值，并保存模型
            self.best_acc = acc
            self.best_loss = loss
            self.save_checkpoint(model)
        elif (loss > self.best_loss) and (acc < self.best_acc):
            # 如果当前损失值大于最佳损失值且准确率小于最佳准确率，计数器加 1
            self.counter += 1
            print(
                f"EarlyStopping counter: {self.counter} out of {self.patience}"
            )
            if self.counter >= self.patience:
                # 如果计数器达到耐心值，设置提前停止训练的标志
                self.early_stop = True
        else:
            if (loss <= self.best_loss) and (acc >= self.best_acc):
                # 如果当前损失值小于等于最佳损失值且准确率大于等于最佳准确率，保存模型
                self.save_checkpoint(model)
            # 更新最佳损失值和最佳准确率
            self.best_loss = np.min((loss, self.best_loss))
            self.best_acc = np.max((acc, self.best_acc))
            # 重置计数器
            self.counter = 0
        return self.early_stop

    def save_checkpoint(self, model):
        """
        当验证集损失值下降时保存模型。

        参数
        ----------
        model : torch.nn.Module
            模型
        """
        # 保存模型的状态字典
        torch.save(model.state_dict(), self.filename)

    def load_checkpoint(self, model):
        """
        加载最新的检查点（保存的模型参数）。

        参数
        ----------
        model : torch.nn.Module
            模型
        """
        # 加载模型的状态字典
        model.load_state_dict(torch.load(self.filename, weights_only=False))
