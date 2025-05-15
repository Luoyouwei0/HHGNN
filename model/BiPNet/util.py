# 导入 numpy 库，用于科学计算
import numpy as np
# 从 scipy.sparse 模块导入 csr_matrix 类，用于创建和操作压缩稀疏行矩阵
from scipy.sparse import csr_matrix
# 从 operator 模块导入 itemgetter 函数，用于获取对象的指定项
from operator import itemgetter

# 定义一个函数，用于生成数据的掩码矩阵
def data_masks(all_sessions, n_node):
    # 初始化三个列表，分别用于存储压缩稀疏行矩阵的索引指针、索引和数据
    indptr, indices, data = [], [], []
    # 索引指针列表的第一个元素为 0
    indptr.append(0)
    # 遍历所有会话
    for j in range(len(all_sessions)):
        # 统计会话中不同的 item，去重并按照 item_id 排序
        session = np.unique(all_sessions[j])
        # 获取当前会话的长度
        length = len(session)
        # 获取索引指针列表的最后一个元素
        s = indptr[-1]
        # 将当前会话长度累加到索引指针列表中
        indptr.append((s + length))
        # 遍历当前会话中的每个 item
        for i in range(length):
            # 将 item_id 减 1 后添加到索引列表中
            indices.append(session[i]-1)
            # 将 item 在会话内的权重（全部为 1）添加到数据列表中
            data.append(1)
    # 注释说明 indptr、indices 和 DBLP 的含义
    # indptr:session长度累加和; indices:item_id 减1, 由每个session内item组成; DBLP:item在session内的权重，全部为1.
    # 创建压缩稀疏行矩阵，形状为 (会话数量, 节点数量)
    matrix = csr_matrix((data, indices, indptr), shape=(len(all_sessions), n_node))
    # 注释说明矩阵的含义，10000 * 6558 #sessions * #items H in paper 稀疏矩阵存储
    return matrix

# 定义一个函数，用于根据给定的数据列表生成简单的掩码矩阵
def data_easy_masks(data_l, n_row, n_col):
    # 从数据列表中提取数据、索引和索引指针
    data, indices, indptr  = data_l[0], data_l[1], data_l[2]
    # 创建压缩稀疏行矩阵，形状为 (指定的行数, 指定的列数)
    matrix = csr_matrix((data, indices, indptr), shape=(n_row, n_col))
    # 注释说明矩阵的含义，10000 * 6558 #sessions * #items H in paper 稀疏矩阵存储
    return matrix

# 定义一个函数，用于将训练集分割为训练集和验证集
def split_validation(train_set, valid_portion):
    # 从训练集中解包出训练数据和训练标签
    train_set_x, train_set_y = train_set
    # 获取训练数据的样本数量
    n_samples = len(train_set_x)
    # 生成一个从 0 到 n_samples-1 的整数数组，用于索引训练数据
    sidx = np.arange(n_samples, dtype='int32')
    # 随机打乱索引数组的顺序
    np.random.shuffle(sidx)
    # 计算训练集的样本数量
    n_train = int(np.round(n_samples * (1. - valid_portion)))
    # 根据打乱后的索引数组，提取验证集的训练数据
    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    # 根据打乱后的索引数组，提取验证集的训练标签
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
    # 根据打乱后的索引数组，提取训练集的训练数据
    train_set_x = [train_set_x[s] for s in sidx[:n_train]]
    # 根据打乱后的索引数组，提取训练集的训练标签
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]
    # 返回分割后的训练集和验证集
    return (train_set_x, train_set_y), (valid_set_x, valid_set_y)

# 定义一个 Data 类，用于处理和管理数据
class Data():
    # 类的构造函数，初始化数据对象
    def __init__(self, data, shuffle=False, n_node=None, n_price=None, n_category=None, n_brand=None):
        # 注释说明数据的格式
        # data_formate: sessions, price_seq, matrix_session_item, matrix_session_price, matrix_session_category, matrix_session_brand, matrix_pv, matrix_pb, matrix_pc, matrix_bv, matrix_bc, matrix_cv, labels
        # 将会话数据转换为 numpy 数组
        self.raw = np.asarray(data[0])  # sessions, item_seq
        # 注释说明以下代码被注释掉的原因
        # self.raw = self.raw[:-1]
        # 将价格序列数据转换为 numpy 数组
        self.price_raw = np.asarray(data[1])  # price_seq
        # 注释说明以下代码被注释掉的原因
        # self.price_raw = self.price_raw[:-1]

        # 根据给定的数据列表生成会话-物品的掩码矩阵
        H_T = data_easy_masks(data[2], len(data[0]), n_node)  # 10000 * 6558 #sessions * #items H_T in paper 稀疏矩阵存储
        # 计算会话-物品矩阵的转置矩阵，并进行归一化处理
        BH_T = H_T.T.multiply(1.0 / H_T.sum(axis=1).reshape(1, -1))
        BH_T = BH_T.T
        # 获取会话-物品矩阵的转置矩阵
        H = H_T.T
        # 计算物品-会话矩阵的转置矩阵，并进行归一化处理
        DH = H.T.multiply(1.0 / H.sum(axis=1).reshape(1, -1))
        DH = DH.T
        # 计算物品-物品的邻接矩阵
        DHBH_T = np.dot(DH, BH_T)  # adjacent matrix of item to item (#item * #item)

        # 根据给定的数据列表生成会话-价格的掩码矩阵
        H_p_T = data_easy_masks(data[3], len(data[0]), n_price)  # #sessions * #price
        # 计算会话-价格矩阵的转置矩阵，并进行归一化处理
        BH_p_T = H_p_T.T.multiply(1.0 / H_p_T.sum(axis=1).reshape(1, -1))
        BH_p_T = BH_p_T.T
        # 获取会话-价格矩阵的转置矩阵
        H_p = H_p_T.T
        # 计算价格-会话矩阵的转置矩阵，并进行归一化处理
        DH_p = H_p.T.multiply(1.0 / H_p.sum(axis=1).reshape(1, -1))
        DH_p = DH_p.T
        # 计算价格-价格的邻接矩阵
        DHBH_p_T = np.dot(DH_p, BH_p_T)  # adjacent matrix of price to price (#price * #price)

        # 根据给定的数据列表生成会话-类别的掩码矩阵
        H_c_T = data_easy_masks(data[4], len(data[0]), n_category)  # #sessions * #category
        # 计算会话-类别矩阵的转置矩阵，并进行归一化处理
        BH_c_T = H_c_T.T.multiply(1.0 / H_c_T.sum(axis=1).reshape(1, -1))
        BH_c_T = BH_c_T.T
        # 获取会话-类别矩阵的转置矩阵
        H_c = H_c_T.T
        # 计算类别-会话矩阵的转置矩阵，并进行归一化处理
        DH_c = H_c.T.multiply(1.0 / H_c.sum(axis=1).reshape(1, -1))
        DH_c = DH_c.T
        # 计算类别-类别的邻接矩阵
        DHBH_c_T = np.dot(DH_c, BH_c_T)  # adjacent matrix of price to price (#price * #price)

        # 根据给定的数据列表生成会话-品牌的掩码矩阵
        H_b_T = data_easy_masks(data[5], len(data[0]), n_brand)  # #sessions * #brand
        # 计算会话-品牌矩阵的转置矩阵，并进行归一化处理
        BH_b_T = H_b_T.T.multiply(1.0 / H_b_T.sum(axis=1).reshape(1, -1))
        BH_b_T = BH_b_T.T
        # 获取会话-品牌矩阵的转置矩阵
        H_b = H_b_T.T
        # 计算品牌-会话矩阵的转置矩阵，并进行归一化处理
        DH_b = H_b.T.multiply(1.0 / H_b.sum(axis=1).reshape(1, -1))
        DH_b = DH_b.T
        # 计算品牌-品牌的邻接矩阵
        DHBH_b_T = np.dot(DH_b, BH_b_T)  # adjacent matrix of price to price (#price * #price)

        # 根据给定的数据列表生成价格-物品的掩码矩阵
        H_pv = data_easy_masks(data[6], n_price, n_node)  # 稀疏矩阵存储
        # 价格-物品的邻接矩阵
        BH_pv = H_pv  # adjacent matrix of price to item (#price * #item)
        # 物品-价格的邻接矩阵
        BH_vp = H_pv.T

        # 根据给定的数据列表生成价格-品牌的掩码矩阵
        H_pb = data_easy_masks(data[7], n_price, n_brand)  # 稀疏矩阵存储
        # 价格-品牌的邻接矩阵
        BH_pb = H_pb  # adjacent matrix of price to item (#price * #item)
        # 品牌-价格的邻接矩阵
        BH_bp = H_pb.T

        # 根据给定的数据列表生成价格-类别的掩码矩阵
        H_pc = data_easy_masks(data[8], n_price, n_category)  # 稀疏矩阵存储
        # 价格-类别的邻接矩阵
        BH_pc = H_pc
        # 类别-价格的邻接矩阵
        BH_cp = H_pc.T

        # 根据给定的数据列表生成品牌-物品的掩码矩阵
        H_bv = data_easy_masks(data[9], n_brand, n_node)  # 稀疏矩阵存储
        # 品牌-物品的邻接矩阵
        BH_bv = H_bv  # adjacent matrix of price to item (#price * #item)
        # 物品-品牌的邻接矩阵
        BH_vb = H_bv.T

        # 根据给定的数据列表生成品牌-类别的掩码矩阵
        H_bc = data_easy_masks(data[10], n_brand, n_category)  # 稀疏矩阵存储
        # 品牌-类别的邻接矩阵
        BH_bc = H_bc  # adjacent matrix of price to item (#price * #item)
        # 类别-品牌的邻接矩阵
        BH_cb = H_bc.T

        # 根据给定的数据列表生成类别-物品的掩码矩阵
        H_cv = data_easy_masks(data[11], n_category, n_node)  # 稀疏矩阵存储
        # 类别-物品的邻接矩阵
        BH_cv = H_cv
        # 物品-类别的邻接矩阵
        BH_vc = H_cv.T

        # 将物品-物品的邻接矩阵转换为 COO 格式
        self.adjacency = DHBH_T.tocoo()
        # 将价格-价格的邻接矩阵转换为 COO 格式
        self.adjacency_pp = DHBH_p_T.tocoo()
        # 将类别-类别的邻接矩阵转换为 COO 格式
        self.adjacency_cc = DHBH_c_T.tocoo()
        # 将品牌-品牌的邻接矩阵转换为 COO 格式
        self.adjacency_bb = DHBH_b_T.tocoo()

        # 将价格-物品的邻接矩阵转换为 COO 格式
        self.adjacency_pv = BH_pv.tocoo()
        # 将价格-类别的邻接矩阵转换为 COO 格式
        self.adjacency_pc = BH_pc.tocoo()
        # 将价格-品牌的邻接矩阵转换为 COO 格式
        self.adjacency_pb = BH_pb.tocoo()

        # 将物品-价格的邻接矩阵转换为 COO 格式
        self.adjacency_vp = BH_vp.tocoo()
        # 将物品-类别的邻接矩阵转换为 COO 格式
        self.adjacency_vc = BH_vc.tocoo()
        # 将物品-品牌的邻接矩阵转换为 COO 格式
        self.adjacency_vb = BH_vb.tocoo()

        # 将类别-价格的邻接矩阵转换为 COO 格式
        self.adjacency_cp = BH_cp.tocoo()
        # 将类别-物品的邻接矩阵转换为 COO 格式
        self.adjacency_cv = BH_cv.tocoo()
        # 将类别-品牌的邻接矩阵转换为 COO 格式
        self.adjacency_cb = BH_cb.tocoo()

        # 将品牌-物品的邻接矩阵转换为 COO 格式
        self.adjacency_bv = BH_bv.tocoo()
        # 将品牌-类别的邻接矩阵转换为 COO 格式
        self.adjacency_bc = BH_bc.tocoo()
        # 将品牌-价格的邻接矩阵转换为 COO 格式
        self.adjacency_bp = BH_bp.tocoo()

        # 存储节点数量
        self.n_node = n_node
        # 存储价格数量
        self.n_price = n_price
        # 存储类别数量
        self.n_category = n_category
        # 存储品牌数量
        self.n_brand = n_brand
        # 将会话的目标标签转换为 numpy 数组
        self.targets = np.asarray(data[12])
        # 存储会话的数量
        self.length = len(self.raw)
        # 存储是否打乱数据的标志
        self.shuffle = shuffle

    # 定义一个方法，用于计算会话之间的重叠度
    def get_overlap(self, sessions):
        # 初始化一个零矩阵，用于存储会话之间的重叠度
        matrix = np.zeros((len(sessions), len(sessions)))
        # 遍历所有会话
        for i in range(len(sessions)):
            # 将当前会话转换为集合，并去除其中的 0
            seq_a = set(sessions[i])
            seq_a.discard(0)
            # 遍历当前会话之后的所有会话
            for j in range(i+1, len(sessions)):
                # 将下一个会话转换为集合，并去除其中的 0
                seq_b = set(sessions[j])
                seq_b.discard(0)
                # 计算两个会话的交集
                overlap = seq_a.intersection(seq_b)
                # 计算两个会话的并集
                ab_set = seq_a | seq_b
                # 计算两个会话的重叠度，并存储在矩阵中
                matrix[i][j] = float(len(overlap))/float(len(ab_set))
                matrix[j][i] = matrix[i][j]
        # 将矩阵的对角线元素设置为 1
        matrix = matrix + np.diag([1.0]*len(sessions))
        # 计算矩阵每一行的元素之和
        degree = np.sum(np.array(matrix), 1)
        # 将每一行的元素之和转换为对角矩阵，并取倒数
        degree = np.diag(1.0/degree)
        # 返回会话重叠度矩阵和度矩阵
        return matrix, degree

    # 定义一个方法，用于生成批次数据的索引切片
    def generate_batch(self, batch_size):
        # 如果需要打乱数据
        if self.shuffle:
            # 生成一个从 0 到会话数量-1 的整数数组，用于索引会话
            shuffled_arg = np.arange(self.length)
            # 随机打乱索引数组的顺序
            np.random.shuffle(shuffled_arg)
            # 根据打乱后的索引数组，重新排列会话数据
            self.raw = self.raw[shuffled_arg]
            # 根据打乱后的索引数组，重新排列价格序列数据
            self.price_raw = self.price_raw[shuffled_arg]
            # 根据打乱后的索引数组，重新排列目标标签数据
            self.targets = self.targets[shuffled_arg]
        # 如果会话数量小于批次大小
        if self.length < batch_size:
            # 计算需要重复的次数
            iterative = int(batch_size / self.length)
            # 初始化一个空列表，用于存储索引切片
            slices = []
            # 将 0 到会话数量-1 的索引重复添加到切片列表中
            slices += list(np.arange(0, self.length)) * iterative
            # 如果批次大小不能被会话数量整除，添加剩余的索引
            if batch_size % self.length != 0:
                slices += list(np.arange(0, batch_size - iterative * self.length))
            # 将切片列表包装在一个列表中
            slices = [slices]
        else:
            # 计算批次的数量
            n_batch = int(self.length / batch_size)
            # 如果会话数量不能被批次大小整除，批次数量加 1
            if self.length % batch_size != 0:
                n_batch += 1
            # 将 0 到批次数量 * 批次大小的索引分割为多个批次的索引切片
            slices = np.split(np.arange(n_batch * batch_size), n_batch)
            # 最后一个批次的索引切片设置为从会话数量-批次大小到会话数量
            slices[-1] = np.arange(self.length-batch_size, self.length)
        # 返回批次数据的索引切片
        return slices

    # 定义一个方法，用于获取指定索引的切片数据
    def get_slice(self, index):
        # 初始化三个空列表，分别用于存储物品序列、物品数量和价格序列
        items, num_node, price_seqs = [], [], []
        # 获取指定索引的会话数据
        inp = self.raw[index]
        # 获取指定索引的价格序列数据
        inp_price = self.price_raw[index]
        # 遍历指定索引的会话数据
        for session in inp:
            # 计算当前会话中非零元素的数量
            num_node.append(len(np.nonzero(session)[0]))
        # 获取所有会话中非零元素数量的最大值
        max_n_node = np.max(num_node)
        # 初始化三个空列表，分别用于存储会话长度、反转的会话物品序列和掩码
        session_len = []
        reversed_sess_item = []
        mask = []
        # 遍历指定索引的会话数据和价格序列数据
        for session, price in zip(inp,inp_price):
            # 将会话数据和价格序列数据转换为列表
            session = list(session)
            price = list(price)
            # 获取当前会话中非零元素的索引
            nonzero_elems = np.nonzero(session)[0]
            # 将当前会话的长度添加到会话长度列表中
            session_len.append([len(nonzero_elems)])
            # 将当前会话填充到最大长度，并添加到物品序列列表中
            items.append(session + (max_n_node - len(nonzero_elems)) * [0])
            # 将当前价格序列填充到最大长度，并添加到价格序列列表中
            price_seqs.append(price + (max_n_node - len(nonzero_elems)) * [0])
            # 生成掩码列表，并添加到掩码列表中
            mask.append([1]*len(nonzero_elems) + (max_n_node - len(nonzero_elems)) * [0])
            # 将当前会话反转并填充到最大长度，并添加到反转的会话物品序列列表中
            reversed_sess_item.append(list(reversed(session)) + (max_n_node - len(nonzero_elems)) * [0])
        # 返回指定索引的目标标签减 1、会话长度、物品序列、反转的会话物品序列、掩码和价格序列
        return self.targets[index]-1, session_len,items, reversed_sess_item, mask, price_seqs


