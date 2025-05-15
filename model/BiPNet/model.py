import datetime
import math
import numpy as np
import torch
from torch import nn, backends
from torch.nn import Module, Parameter
import torch.nn.functional as F
import torch.sparse
from scipy.sparse import coo
import time


def trans_to_cuda(variable):

    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable


def trans_to_cpu(variable):

    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable


class HyperConv(Module):
    def __init__(self, layers, dataset, emb_size, n_node, n_price, n_category, n_brand):
        super(HyperConv, self).__init__()
        self.emb_size = emb_size  # 嵌入向量的维度
        self.layers = layers  # 超图卷积的层数
        self.dataset = dataset  # 数据集对象
        self.n_node = n_node  # 节点数量
        self.n_price = n_price  # 价格数量
        self.n_category = n_category  # 类别数量
        self.n_brand = n_brand  # 品牌数量

        self.w_pv = nn.Linear(self.emb_size, self.emb_size)
        self.w_bv = nn.Linear(self.emb_size, self.emb_size)
        self.w_cv = nn.Linear(self.emb_size, self.emb_size)

        self.w_bp = nn.Linear(self.emb_size, self.emb_size)
        self.w_cp = nn.Linear(self.emb_size, self.emb_size)
        self.w_vp = nn.Linear(self.emb_size, self.emb_size)

        self.w_pb = nn.Linear(self.emb_size, self.emb_size)
        self.w_cb = nn.Linear(self.emb_size, self.emb_size)
        self.w_vb = nn.Linear(self.emb_size, self.emb_size)

        self.w_pc = nn.Linear(self.emb_size, self.emb_size)
        self.w_bc = nn.Linear(self.emb_size, self.emb_size)
        self.w_vc = nn.Linear(self.emb_size, self.emb_size)


        self.tran_pv = nn.Linear(self.emb_size, self.emb_size)
        self.tran_pc = nn.Linear(self.emb_size, self.emb_size)
        self.tran_pb = nn.Linear(self.emb_size, self.emb_size)

        self.tran_cv = nn.Linear(self.emb_size, self.emb_size)
        self.tran_cp = nn.Linear(self.emb_size, self.emb_size)
        self.tran_cb = nn.Linear(self.emb_size, self.emb_size)

        self.tran_bv = nn.Linear(self.emb_size, self.emb_size)
        self.tran_bp = nn.Linear(self.emb_size, self.emb_size)
        self.tran_bc = nn.Linear(self.emb_size, self.emb_size)

        self.tran_pv2 = nn.Linear(self.emb_size, self.emb_size)
        self.tran_pc2 = nn.Linear(self.emb_size, self.emb_size)
        self.tran_pb2 = nn.Linear(self.emb_size, self.emb_size)

        self.tran_cv2 = nn.Linear(self.emb_size, self.emb_size)
        self.tran_cp2 = nn.Linear(self.emb_size, self.emb_size)
        self.tran_cb2 = nn.Linear(self.emb_size, self.emb_size)

        self.tran_bv2 = nn.Linear(self.emb_size, self.emb_size)
        self.tran_bp2 = nn.Linear(self.emb_size, self.emb_size)
        self.tran_bc2 = nn.Linear(self.emb_size, self.emb_size)

        self.a_pb = nn.Linear(self.emb_size, self.emb_size)
        self.a_pc = nn.Linear(self.emb_size, self.emb_size)
        self.a_bc = nn.Linear(self.emb_size, self.emb_size)
        self.a_bp = nn.Linear(self.emb_size, self.emb_size)
        self.a_cb = nn.Linear(self.emb_size, self.emb_size)
        self.a_cp = nn.Linear(self.emb_size, self.emb_size)

        self.b_pb = nn.Linear(self.emb_size, self.emb_size)
        self.b_pc = nn.Linear(self.emb_size, self.emb_size)
        self.b_bc = nn.Linear(self.emb_size, self.emb_size)
        self.b_bp = nn.Linear(self.emb_size, self.emb_size)
        self.b_cb = nn.Linear(self.emb_size, self.emb_size)
        self.b_cp = nn.Linear(self.emb_size, self.emb_size)

        # self.mat_v = nn.Parameter(torch.Tensor(self.n_node, self.emb_size))
        self.mat_pc = nn.Parameter(torch.Tensor(self.n_price, 1))
        self.mat_pb = nn.Parameter(torch.Tensor(self.n_price, 1))
        self.mat_pv = nn.Parameter(torch.Tensor(self.n_price, 1))

        self.mat_cp = nn.Parameter(torch.Tensor(self.n_category, 1))
        self.mat_cb = nn.Parameter(torch.Tensor(self.n_category, 1))
        self.mat_cv = nn.Parameter(torch.Tensor(self.n_category, 1))

        self.mat_bp = nn.Parameter(torch.Tensor(self.n_brand, 1))
        self.mat_bc = nn.Parameter(torch.Tensor(self.n_brand, 1))
        self.mat_bv = nn.Parameter(torch.Tensor(self.n_brand, 1))

        # self.mat_bc = nn.Parameter(torch.Tensor(self.n_brand, self.emb_size))

        self.a_i_g = nn.Linear(self.emb_size, self.emb_size)
        self.b_i_g = nn.Linear(self.emb_size, self.emb_size)

        self.w_v_1 = nn.Linear(self.emb_size * 4, self.emb_size)
        self.w_v_11 = nn.Linear(self.emb_size * 2, self.emb_size)
        self.w_v_2 = nn.Linear(self.emb_size * 1, self.emb_size)
        self.w_v_3 = nn.Linear(self.emb_size * 1, self.emb_size)
        self.w_v_4 = nn.Linear(self.emb_size * 1, self.emb_size)

        self.w_p_1 = nn.Linear(self.emb_size * 4, self.emb_size)
        self.w_p_11 = nn.Linear(self.emb_size * 2, self.emb_size)
        self.w_p_2 = nn.Linear(self.emb_size * 1, self.emb_size)
        self.w_p_3 = nn.Linear(self.emb_size * 1, self.emb_size)
        self.w_p_4 = nn.Linear(self.emb_size * 1, self.emb_size)

        self.w_c_1 = nn.Linear(self.emb_size * 4, self.emb_size)
        self.w_c_11 = nn.Linear(self.emb_size * 2, self.emb_size)
        self.w_c_2 = nn.Linear(self.emb_size * 1, self.emb_size)
        self.w_c_3 = nn.Linear(self.emb_size * 1, self.emb_size)
        self.w_c_4 = nn.Linear(self.emb_size * 1, self.emb_size)

        self.w_b_1 = nn.Linear(self.emb_size * 4, self.emb_size)
        self.w_b_11 = nn.Linear(self.emb_size * 2, self.emb_size)
        self.w_b_2 = nn.Linear(self.emb_size * 1, self.emb_size)
        self.w_b_3 = nn.Linear(self.emb_size * 1, self.emb_size)
        self.w_b_4 = nn.Linear(self.emb_size * 1, self.emb_size)

        self.dropout10 = nn.Dropout(0.1)
        self.dropout20 = nn.Dropout(0.2)
        self.dropout30 = nn.Dropout(0.3)
        self.dropout40 = nn.Dropout(0.4)
        self.dropout50 = nn.Dropout(0.5)
        self.dropout60 = nn.Dropout(0.6)
        self.dropout70 = nn.Dropout(0.7)
        self.dropout90 = nn.Dropout(0.9)


    def forward(self, adjacency, adjacency_pp, adjacency_cc, adjacency_bb, adjacency_vp, adjacency_vc, adjacency_vb,
                adjacency_pv, adjacency_pc, adjacency_pb, adjacency_cv, adjacency_cp, adjacency_cb, adjacency_bv,
                adjacency_bp, adjacency_bc, embedding, pri_emb, cate_emb, bra_emb):
        """
        前向传播方法，通过多层超图卷积更新物品、价格、类别和品牌的嵌入向量。

        参数:
        adjacency (sparse matrix): 物品之间的邻接矩阵。
        adjacency_pp (sparse matrix): 价格之间的邻接矩阵。
        adjacency_cc (sparse matrix): 类别之间的邻接矩阵。
        adjacency_bb (sparse matrix): 品牌之间的邻接矩阵。
        adjacency_vp (sparse matrix): 物品到价格的邻接矩阵。
        adjacency_vc (sparse matrix): 物品到类别的邻接矩阵。
        adjacency_vb (sparse matrix): 物品到品牌的邻接矩阵。
        adjacency_pv (sparse matrix): 价格到物品的邻接矩阵。
        adjacency_pc (sparse matrix): 价格到类别的邻接矩阵。
        adjacency_pb (sparse matrix): 价格到品牌的邻接矩阵。
        adjacency_cv (sparse matrix): 类别到物品的邻接矩阵。
        adjacency_cp (sparse matrix): 类别到价格的邻接矩阵。
        adjacency_cb (sparse matrix): 类别到品牌的邻接矩阵。
        adjacency_bv (sparse matrix): 品牌到物品的邻接矩阵。
        adjacency_bp (sparse matrix): 品牌到价格的邻接矩阵。
        adjacency_bc (sparse matrix): 品牌到类别的邻接矩阵。
        embedding (torch.Tensor): 物品的初始嵌入向量。
        pri_emb (torch.Tensor): 价格的初始嵌入向量。
        cate_emb (torch.Tensor): 类别的初始嵌入向量。
        bra_emb (torch.Tensor): 品牌的初始嵌入向量。

        返回:
        torch.Tensor: 经过多层卷积更新后的物品嵌入向量。
        torch.Tensor: 经过多层卷积更新后的价格嵌入向量。
        """
        for i in range(self.layers):
            item_embeddings = self.inter_gate3(
                self.w_v_1, self.w_v_11, self.w_v_2, self.w_v_3, self.w_v_4, embedding,
                self.get_embedding(adjacency_vp, pri_emb),
                self.get_embedding(adjacency_vc, cate_emb),
                self.get_embedding(adjacency_vb, bra_emb)) + self.get_embedding(adjacency, embedding)

            price_embeddings = self.inter_gate3(
                self.w_p_1, self.w_p_11, self.w_p_2, self.w_p_3, self.w_p_4, pri_emb,
                self.intra_gate2(adjacency_pv, self.mat_pv, self.tran_pv, self.tran_pv2, pri_emb, embedding),
                self.intra_gate2(adjacency_pc, self.mat_pc, self.tran_pc, self.tran_pc2, pri_emb, cate_emb),
                self.intra_gate2(adjacency_pb, self.mat_pb, self.tran_pb, self.tran_pb2, pri_emb,
                                 bra_emb)) + self.get_embedding(adjacency_pp, pri_emb)

            category_embeddings = self.inter_gate3(
                self.w_c_1, self.w_c_11, self.w_c_2, self.w_c_3, self.w_c_4, cate_emb,
                self.intra_gate2(adjacency_cp, self.mat_cp, self.tran_cp, self.tran_cp2, cate_emb, pri_emb),
                self.intra_gate2(adjacency_cv, self.mat_cv, self.tran_cv, self.tran_cv2, cate_emb, embedding),
                self.intra_gate2(adjacency_cb, self.mat_cb, self.tran_cb, self.tran_cb2, cate_emb,
                                 bra_emb)) + self.get_embedding(adjacency_cc, cate_emb)
            brand_embeddings = self.inter_gate3(
                self.w_b_1, self.w_b_11, self.w_b_2, self.w_b_3, self.w_b_4, bra_emb,
                self.intra_gate2(adjacency_bp, self.mat_bp, self.tran_bp, self.tran_bp2, bra_emb, pri_emb),
                self.intra_gate2(adjacency_bc, self.mat_bc, self.tran_bc, self.tran_bc2, bra_emb, cate_emb),
                self.intra_gate2(adjacency_bv, self.mat_bv, self.tran_bv, self.tran_bv2, bra_emb, embedding)
            ) + self.get_embedding(adjacency_bb, bra_emb)
            embedding = item_embeddings
            pri_emb = price_embeddings
            cate_emb = category_embeddings
            bra_emb = brand_embeddings

        return item_embeddings, price_embeddings


    def get_embedding(self, adjacency, embedding):
        """
        根据邻接矩阵和输入的嵌入向量计算新的嵌入向量。

        参数:
        adjacency (sparse matrix): 稀疏邻接矩阵，描述节点之间的连接关系。
        embedding (torch.Tensor): 输入的嵌入向量，通常表示节点的特征。

        返回:
        torch.Tensor: 经过稀疏矩阵乘法计算得到的新的嵌入向量。
        """
        # 从邻接矩阵中提取非零元素的值
        values = adjacency.data
        # 将邻接矩阵的行和列索引堆叠成一个二维数组
        indices = np.vstack((adjacency.row, adjacency.col))
        # 将索引数组转换为 torch 的 LongTensor 类型
        i = torch.LongTensor(indices)
        # 将非零元素的值转换为 torch 的 FloatTensor 类型
        v = torch.FloatTensor(values)

        # 获取邻接矩阵的形状
        shape = adjacency.shape
        # 使用索引、值和形状创建一个稀疏的 FloatTensor
        adjacency = torch.sparse.FloatTensor(i, v, torch.Size(shape))
        # 保存输入的嵌入向量
        embs = embedding
        # 将邻接矩阵转移到 GPU 上（如果可用），并与嵌入向量进行稀疏矩阵乘法
        item_embeddings = torch.sparse.mm(trans_to_cuda(adjacency), embs)
        return item_embeddings

    def intra_gate(self, adjacency, mat_v, trans1, trans2, embedding1, embedding2):
        """
        使用 v_attention 方法获取特定类型的嵌入向量，然后通过门控机制得到最终的类型嵌入向量。

        参数:
        adjacency (sparse matrix): 稀疏邻接矩阵，描述节点之间的连接关系。
        mat_v (torch.Tensor): 可训练的参数矩阵，用于计算注意力权重。
        trans1 (nn.Module): 线性变换模块，此处代码中暂时未使用。
        trans2 (nn.Module): 线性变换模块，此处代码中暂时未使用。
        embedding1 (torch.Tensor): 输入的嵌入向量 1。
        embedding2 (torch.Tensor): 输入的嵌入向量 2，用于计算注意力加权后的嵌入。

        返回:
        torch.Tensor: 经过注意力机制和 dropout 处理后的嵌入向量。
        """
        # v_attention to get embedding of type, and then gate to get final type embedding
        # 从邻接矩阵中提取非零元素的值
        values = adjacency.data
        # 将邻接矩阵的行和列索引堆叠成一个二维数组
        indices = np.vstack((adjacency.row, adjacency.col))
        # 将索引数组转换为 torch 的 LongTensor 类型
        i = torch.LongTensor(indices)
        # 将非零元素的值转换为 torch 的 FloatTensor 类型
        v = torch.FloatTensor(values)
        # 获取邻接矩阵的形状
        shape = adjacency.shape
        # 使用索引、值和形状创建一个稀疏的 FloatTensor
        adjacency = torch.sparse.FloatTensor(i, v, torch.Size(shape))
        # 将稀疏矩阵转换为密集矩阵并移动到 GPU 上
        matrix = adjacency.to_dense().cuda()
        # tran_emb2 = trans1(embedding2)
        # 将 mat_v 扩展到与嵌入向量相同的维度
        mat_v = mat_v.expand(mat_v.shape[0], self.emb_size)
        # 计算 mat_v 与 embedding2 转置的矩阵乘法，得到注意力分数
        alpha = torch.mm(mat_v, torch.transpose(embedding2, 1, 0))
        # 对注意力分数应用 Softmax 函数，得到注意力权重
        alpha = torch.nn.Softmax(dim=1)(alpha)
        # 将注意力权重与邻接矩阵逐元素相乘
        alpha = alpha * matrix
        # 计算每行注意力权重的总和，并扩展到与 alpha 相同的形状，加上一个小的常数防止除零错误
        sum_alpha_row = torch.sum(alpha, 1).unsqueeze(1).expand_as(alpha) + 1e-8
        # 对注意力权重进行归一化处理
        alpha = alpha / sum_alpha_row

        # 计算注意力加权后的嵌入向量
        type_embs = torch.mm(alpha, embedding2)

        item_embeddings = type_embs
        # 对最终的嵌入向量应用 dropout 操作，丢弃 70% 的元素
        return self.dropout70(item_embeddings)
    def intra_gate2(self, adjacency, mat_v, trans1, trans2, embedding1, embedding2):
        # v_attention to get embedding of type, and then gate to get final type embedding
        values = adjacency.data
        indices = np.vstack((adjacency.row, adjacency.col))
        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = adjacency.shape
        adjacency = torch.sparse.FloatTensor(i, v, torch.Size(shape))
        matrix = adjacency.to_dense().cuda()

        tran_emb2 = trans1(embedding2)

        alpha = torch.mm(embedding1, torch.transpose(tran_emb2, 1, 0))
        alpha = torch.nn.Softmax(dim=1)(alpha)
        alpha = alpha * matrix
        sum_alpha_row = torch.sum(alpha, 1).unsqueeze(1).expand_as(alpha) + 1e-8
        alpha = alpha / sum_alpha_row

        type_embs = torch.mm(alpha, embedding2)
        item_embeddings = type_embs
        return item_embeddings
    def inter_gate(self, a_o_g, b_o_g1, b_o_g2, emb_mat1, emb_mat2, emb_mat3):
        all_emb1 = self.dropout70(torch.cat([emb_mat1, emb_mat2, emb_mat3], 1))
        # all_emb2 = torch.cat([emb_mat1, emb_mat3], 1)
        gate1 = torch.sigmoid(a_o_g(all_emb1) + b_o_g1(emb_mat2) + b_o_g2(emb_mat3))
        # gate2 = torch.sigmoid(a_o_g(all_emb2) + b_o_g2(emb_mat3))
        h_embedings = emb_mat1 + gate1 * emb_mat2 + (1 - gate1) * emb_mat3


        return h_embedings

    def inter_gate3(self, w1, w11, w2, w3, w4, emb1, emb2, emb3, emb4):
        # 4 to 1
        all_emb = torch.cat([emb1, emb2, emb3, emb4], 1)

        gate1 = torch.tanh(w1(all_emb) + w2(emb2))
        gate2 = torch.tanh(w1(all_emb) + w3(emb3))
        gate3 = torch.tanh(w1(all_emb) + w4(emb4))
        # gate2 = torch.sigmoid(a_o_g(all_emb2) + b_o_g2(emb_mat3))
        h_embedings = emb1 + gate1 * emb2 + gate2 * emb3 + gate3 * emb4
        return h_embedings


    def intra_att(self, adjacency, trans1, trans2, embedding1, embedding2):
        # mlp 映射到相同空间，然后计算cosine相似度，确定attention值
        values = adjacency.data
        indices = np.vstack((adjacency.row, adjacency.col))
        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = adjacency.shape
        adjacency = torch.sparse.FloatTensor(i, v, torch.Size(shape))
        tran_m1 = trans1(embedding1)
        tran_m2 = trans2(embedding2)
        tran_m2 = torch.transpose(tran_m2, 1, 0)
        atten = torch.matmul(tran_m1, tran_m2)
        sum2_m1 = torch.sum(torch.mul(tran_m1, tran_m1),1)
        sum2_m2 = torch.sum(torch.mul(torch.transpose(tran_m2, 1, 0), torch.transpose(tran_m2, 1, 0)),1)
        fenmu_m1 = torch.sqrt(sum2_m1).unsqueeze(1)
        fenmu_m2 = torch.sqrt(sum2_m2).unsqueeze(1)
        fenmu = torch.matmul(fenmu_m1, torch.transpose(fenmu_m2, 1, 0)) + 1e-8
        atten = torch.div(atten, fenmu)
        atten = torch.nn.Softmax(dim=1)(atten)
        atten = torch.mul(adjacency.to_dense().cuda(),atten) + 1e-8
        atten = torch.div(atten, torch.sum(atten, 1).unsqueeze(1).expand_as(atten))
        embs = embedding2
        item_embeddings = torch.mm(atten, embs)
        return item_embeddings

    def inter_att(self, v_mat, emb_mat1, emb_mat2, emb_mat3=None):
        if emb_mat3 is not None:
            alpha1 = torch.mul(v_mat, emb_mat1)
            alpha1 = torch.sum(alpha1, 1).unsqueeze(1)
            alpha2 = torch.mul(v_mat, emb_mat2)
            alpha2 = torch.sum(alpha2, 1).unsqueeze(1)
            alpha3 = torch.mul(v_mat, emb_mat3)
            alpha3 = torch.sum(alpha3, 1).unsqueeze(1)
            alpha = torch.cat([alpha1, alpha2, alpha3], 1)
            alpha = torch.nn.Softmax(dim=1)(alpha).permute(1, 0)
            alpha1 = alpha[0]
            alpha2 = alpha[1]
            alpha3 = alpha[2]
            weight_embs = alpha1.unsqueeze(1).expand_as(emb_mat1) * emb_mat1 + alpha2.unsqueeze(1).expand_as(
                emb_mat2) * emb_mat2 + alpha3.unsqueeze(1).expand_as(emb_mat3) * emb_mat3
        else:
            alpha1 = torch.mul(v_mat, emb_mat1)
            alpha1 = torch.sum(alpha1, 1).unsqueeze(1)
            alpha2 = torch.mul(v_mat, emb_mat2)
            alpha2 = torch.sum(alpha2, 1).unsqueeze(1)
            # alpha3 = torch.mul(v_mat, emb_mat3)
            # alpha3 = torch.sum(alpha3, 1).unsqueeze(1)
            alpha = torch.cat([alpha1, alpha2], 1)
            alpha = torch.nn.Softmax(dim=1)(alpha).permute(1, 0)
            alpha1 = alpha[0]
            alpha2 = alpha[1]
            # alpha3 = alpha[2]
            weight_embs = alpha1.unsqueeze(1).expand_as(emb_mat1) * emb_mat1 + alpha2.unsqueeze(1).expand_as(
                emb_mat2) * emb_mat2

        return weight_embs

class DHCN(Module):
 
    def __init__(self, adjacency, adjacency_pp, adjacency_cc, adjacency_bb, adjacency_vp, adjacency_vc, adjacency_vb, adjacency_pv, adjacency_pc, adjacency_pb, adjacency_cv, adjacency_cp, adjacency_cb, adjacency_bv, adjacency_bp, adjacency_bc, n_node, n_price, n_category, n_brand, lr, layers, l2, beta, dataset, num_heads=4, emb_size=100, batch_size=100):
        # 调用父类的构造函数
        super(DHCN, self).__init__()
        # 嵌入向量的维度
        self.emb_size = emb_size
        # 批量大小
        self.batch_size = batch_size
        # 节点的数量
        self.n_node = n_node
        # 价格的数量
        self.n_price = n_price
        # 类别的数量
        self.n_category = n_category
        # 品牌的数量
        self.n_brand = n_brand
        # L2 正则化系数
        self.L2 = l2
        # 学习率
        self.lr = lr
        # 超图卷积的层数
        self.layers = layers
        # 自定义的超参数
        self.beta = beta

        # 存储各种邻接矩阵
        self.adjacency = adjacency
        self.adjacency_pp = adjacency_pp
        self.adjacency_cc = adjacency_cc
        self.adjacency_bb = adjacency_bb

        self.adjacency_vp = adjacency_vp
        self.adjacency_vc = adjacency_vc
        self.adjacency_vb = adjacency_vb

        self.adjacency_pv = adjacency_pv
        self.adjacency_pc = adjacency_pc
        self.adjacency_pb = adjacency_pb

        self.adjacency_cv = adjacency_cv
        self.adjacency_cp = adjacency_cp
        self.adjacency_cb = adjacency_cb

        self.adjacency_bv = adjacency_bv
        self.adjacency_bp = adjacency_bp
        self.adjacency_bc = adjacency_bc

        # 定义各种嵌入层
        # 物品嵌入层
        self.embedding = nn.Embedding(self.n_node, self.emb_size)
        # 价格嵌入层
        self.price_embedding = nn.Embedding(self.n_price, self.emb_size)
        # 类别嵌入层
        self.category_embedding = nn.Embedding(self.n_category, self.emb_size)
        # 品牌嵌入层
        self.brand_embedding = nn.Embedding(self.n_brand, self.emb_size)

        # 位置嵌入层
        self.pos_embedding = nn.Embedding(2000, self.emb_size)
        # 初始化超图卷积模块
        self.HyperGraph = HyperConv(self.layers, dataset, self.emb_size, self.n_node, self.n_price, self.n_category, self.n_brand)
        # 定义线性层
        self.w_1 = nn.Linear(self.emb_size*2, self.emb_size)
        self.w_price_1 = nn.Linear(self.emb_size * 2, self.emb_size)
        self.w_2 = nn.Linear(self.emb_size, 1)
        # 定义 GLU 层
        self.glu1 = nn.Linear(self.emb_size, self.emb_size)
        self.glu2 = nn.Linear(self.emb_size, self.emb_size, bias=False)
        self.glu3 = nn.Linear(self.emb_size, self.emb_size, bias=False)

        # 自注意力机制部分
        # 检查嵌入维度是否能被头数整除
        if emb_size % num_heads != 0:  # 整除
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (emb_size, num_heads))
        # 自注意力头数
        self.num_heads = num_heads  # 4
        # 每个注意力头的维度
        self.attention_head_size = int(emb_size / num_heads)  # 16
        # 所有注意力头的总维度
        self.all_head_size = int(self.num_heads * self.attention_head_size)
        # 定义查询、键、值的线性层
        self.query = nn.Linear(self.emb_size , self.emb_size )  # 128, 128
        self.key = nn.Linear(self.emb_size , self.emb_size )
        self.value = nn.Linear(self.emb_size , self.emb_size )

        # 定义各种门控机制的线性层
        self.w_p_z = nn.Linear(self.emb_size, self.emb_size, bias=True)
        self.w_p_r = nn.Linear(self.emb_size, self.emb_size, bias=True)
        self.w_p = nn.Linear(self.emb_size, self.emb_size, bias=True)

        self.u_i_z = nn.Linear(self.emb_size, self.emb_size, bias=True)
        self.u_i_r = nn.Linear(self.emb_size, self.emb_size, bias=True)
        self.u_i = nn.Linear(self.emb_size, self.emb_size, bias=True)

        # 门控机制
        self.w_pi_1 = nn.Linear(self.emb_size, self.emb_size, bias=True)
        self.w_pi_2 = nn.Linear(self.emb_size, self.emb_size, bias=True)
        self.w_c_z = nn.Linear(self.emb_size, self.emb_size, bias=True)
        self.u_j_z = nn.Linear(self.emb_size, self.emb_size, bias=True)
        self.w_c_r = nn.Linear(self.emb_size, self.emb_size, bias=True)
        self.u_j_r = nn.Linear(self.emb_size, self.emb_size, bias=True)
        self.w_p = nn.Linear(self.emb_size, self.emb_size, bias=True)
        self.u_p = nn.Linear(self.emb_size, self.emb_size, bias=True)
        self.w_i = nn.Linear(self.emb_size, self.emb_size, bias=True)
        self.u_i = nn.Linear(self.emb_size, self.emb_size, bias=True)

        # 多任务合并部分
        self.merge_w = nn.Linear(self.emb_size, self.emb_size, bias=True)
        self.merge_w1 = nn.Linear(self.emb_size, self.emb_size, bias=True)
        self.merge_w2 = nn.Linear(self.emb_size, self.emb_size, bias=True)
        self.merge_w3 = nn.Linear(self.emb_size, self.emb_size, bias=True)
        self.merge_w4 = nn.Linear(self.emb_size, self.emb_size, bias=True)
        self.merge_w5 = nn.Linear(self.emb_size, self.emb_size, bias=True)
        self.merge_w6 = nn.Linear(self.emb_size, self.emb_size, bias=True)
        self.merge_w7 = nn.Linear(self.emb_size, self.emb_size, bias=True)
        self.merge_w8 = nn.Linear(self.emb_size, self.emb_size, bias=True)
        self.merge_w9 = nn.Linear(self.emb_size, self.emb_size, bias=True)
        self.merge_w10 = nn.Linear(self.emb_size, self.emb_size, bias=True)
        self.merge_w11 = nn.Linear(self.emb_size, self.emb_size, bias=True)
        self.merge_w12 = nn.Linear(self.emb_size, self.emb_size, bias=True)

        # 多任务学习的 MLP 层
        self.mlp_m_p_1 =  nn.Linear(self.emb_size*2, self.emb_size, bias=True)
        self.mlp_m_i_1 = nn.Linear(self.emb_size * 2, self.emb_size, bias=True)

        self.mlp_m_p_2 = nn.Linear(self.emb_size, self.emb_size, bias=True)
        self.mlp_m_i_2 = nn.Linear(self.emb_size, self.emb_size, bias=True)

        # 定义 Dropout 层
        self.dropout = nn.Dropout(0.2)
        self.emb_dropout = nn.Dropout(0.25)

        self.dropout1 = nn.Dropout(0.1)
        self.dropout3 = nn.Dropout(0.3)
        self.dropout5 = nn.Dropout(0.5)
        self.dropout7 = nn.Dropout(0.7)

        # 定义损失函数
        self.loss_function = nn.CrossEntropyLoss()
        # 定义优化器
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        # 初始化模型参数
        self.init_parameters()
    def init_parameters(self):
        # 计算均匀分布的标准差，该标准差基于嵌入维度的平方根的倒数
        stdv = 1.0 / math.sqrt(self.emb_size)
        # 遍历模型的所有可训练参数
        for weight in self.parameters():
            # 使用均匀分布对参数进行初始化，范围是 [-stdv, stdv]
            weight.data.uniform_(-stdv, stdv)


    def generate_sess_emb(self, item_embedding, price_embedding, session_item, price_seqs, session_len, reversed_sess_item, mask):
        """
        生成会话嵌入向量，结合物品嵌入和价格嵌入，通过自注意力机制和多任务学习得到最终的兴趣偏好和价格偏好。

        参数:
        item_embedding (torch.Tensor): 物品的嵌入向量。
        price_embedding (torch.Tensor): 价格的嵌入向量。
        session_item (torch.Tensor): 会话中的物品序列。
        price_seqs (torch.Tensor): 会话中的价格序列。
        session_len (torch.Tensor): 会话的长度。
        reversed_sess_item (torch.Tensor): 反转后的会话物品序列。
        mask (torch.Tensor): 掩码张量，用于屏蔽无效位置。

        返回:
        torch.Tensor: 最终的兴趣偏好嵌入向量。
        torch.Tensor: 最终的价格偏好嵌入向量。
        """
        # 创建一个全零的张量，用于在嵌入向量前添加零向量
        zeros = torch.cuda.FloatTensor(1, self.emb_size).fill_(0)
        # zeros = torch.zeros(1, self.emb_size)
        # 将掩码张量转换为浮点型，并在最后一维添加一个维度
        mask = mask.float().unsqueeze(-1)

        # 在价格嵌入向量前添加零向量
        price_embedding = torch.cat([zeros, price_embedding], 0)
        # 定义一个匿名函数，用于根据索引获取价格嵌入向量
        get_pri = lambda i: price_embedding[price_seqs[i]]
        # 初始化一个全零的张量，用于存储价格序列的嵌入向量
        seq_pri = torch.cuda.FloatTensor(self.batch_size, list(price_seqs.shape)[1], self.emb_size).fill_(0)
        # seq_h = torch.zeros(self.batch_size, list(reversed_sess_item.shape)[1], self.emb_size)
        # 遍历价格序列，填充 seq_pri 张量
        for i in torch.arange(price_seqs.shape[0]):
            seq_pri[i] = get_pri(i)

        # 为价格嵌入向量添加位置编码
        seq_h = torch.cuda.FloatTensor(self.batch_size, list(reversed_sess_item.shape)[1], self.emb_size).fill_(0)
        # 获取序列的长度
        len = seq_h.shape[1]
        # 获取位置嵌入向量
        pos_emb = self.pos_embedding.weight[:len]
        # 复制位置嵌入向量，使其维度与 seq_pri 匹配
        pos_emb = pos_emb.unsqueeze(0).repeat(self.batch_size, 1, 1)
        # 将位置嵌入向量和价格嵌入向量拼接，并通过线性层进行变换
        seq_pri = self.w_price_1(torch.cat([pos_emb, seq_pri], -1))

        # 使用自注意力机制获取价格偏好
        # 调整掩码张量的维度，用于自注意力机制的掩码
        attention_mask = mask.permute(0,2,1).unsqueeze(1)  # [bs, 1, 1, seqlen]
        # 将掩码值转换为合适的形式，用于屏蔽无效位置
        attention_mask = (1.0 - attention_mask) * -10000.0

        # 通过查询、键、值线性层对价格序列嵌入向量进行变换
        mixed_query_layer = self.query(seq_pri)  # [bs, seqlen, hid_size]
        mixed_key_layer = self.key(seq_pri)  # [bs, seqlen, hid_size]
        mixed_value_layer = self.value(seq_pri)  # [bs, seqlen, hid_size]

        # 计算每个注意力头的维度
        attention_head_size = int(self.emb_size / self.num_heads)
        # 调整查询、键、值张量的维度，以适应多头注意力机制
        query_layer = self.transpose_for_scores(mixed_query_layer, attention_head_size)  # [bs, 8, seqlen, 16]
        key_layer = self.transpose_for_scores(mixed_key_layer, attention_head_size)
        value_layer = self.transpose_for_scores(mixed_value_layer, attention_head_size)  # [bs, 8, seqlen, 16]
        # 计算查询和键的点积，得到原始的注意力分数
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        # 对注意力分数进行缩放
        attention_scores = attention_scores / math.sqrt(attention_head_size)  # [bs, 8, seqlen, seqlen]
        # 应用掩码到注意力分数上
        attention_scores = attention_scores + attention_mask

        # 对注意力分数应用 Softmax 函数，得到注意力概率
        attention_probs = nn.Softmax(dim=-1)(attention_scores)  # [bs, 8, seqlen, seqlen]
        # 对注意力概率应用 Dropout 操作
        attention_probs = self.dropout(attention_probs)

        # 注意力概率与值张量相乘，得到上下文层
        context_layer = torch.matmul(attention_probs, value_layer)  # [bs, 8, seqlen, 16]
        # 调整上下文层的维度
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()  # [bs, seqlen, 8, 16]
        # 计算新的上下文层形状
        new_context_layer_shape = context_layer.size()[:-2] + (self.emb_size,)  # [bs, seqlen, 128]
        # 调整上下文层的形状，得到自注意力结果
        sa_result = context_layer.view(*new_context_layer_shape)
        # 取最后一个隐藏状态作为价格偏好
        # 生成物品位置张量
        item_pos = torch.tensor(range(1, seq_pri.size()[1] + 1), device='cuda')
        # 扩展物品位置张量，使其维度与价格序列匹配
        item_pos = item_pos.unsqueeze(0).expand_as(price_seqs)

        # 应用掩码到物品位置张量上
        item_pos = item_pos * mask.squeeze(2)
        # 获取每个会话中最后一个物品的位置
        item_last_num = torch.max(item_pos, 1)[0].unsqueeze(1).expand_as(item_pos)
        # 生成最后位置的掩码张量
        last_pos_t = torch.where(item_pos - item_last_num >= 0, torch.tensor([1.0], device='cuda'),
                                 torch.tensor([0.0], device='cuda'))
        # 应用最后位置的掩码到自注意力结果上
        last_interest = last_pos_t.unsqueeze(2).expand_as(sa_result) * sa_result
        # 对最后兴趣张量求和，得到价格偏好
        price_pre = torch.sum(last_interest, 1)
        # 另一种计算价格偏好的方式：对自注意力结果求平均
        # price_pre = torch.div(torch.sum(sa_result, 1), session_len)

        # 在物品嵌入向量前添加零向量
        item_embedding = torch.cat([zeros, item_embedding], 0)
        # 定义一个匿名函数，用于根据索引获取物品嵌入向量
        # get = lambda i: item_embedding[session_item[i]]
        # seq_h = torch.cuda.FloatTensor(self.batch_size, list(session_item.shape)[1], self.emb_size).fill_(0)
        get = lambda i: item_embedding[reversed_sess_item[i]]
        # 初始化一个全零的张量，用于存储物品序列的嵌入向量
        seq_h = torch.cuda.FloatTensor(self.batch_size, list(reversed_sess_item.shape)[1], self.emb_size).fill_(0)

        # seq_h = torch.zeros(self.batch_size, list(reversed_sess_item.shape)[1], self.emb_size)
        # 遍历会话物品序列，填充 seq_h 张量
        for i in torch.arange(session_item.shape[0]):
            seq_h[i] = get(i)
        # 取最后一个物品的嵌入向量
        last_price = last_pos_t.unsqueeze(2).expand_as(seq_h) * seq_h
        # 对最后价格张量求和，得到最后一个物品的嵌入向量
        hl = torch.sum(last_price, 1)

        # 计算物品序列嵌入向量的平均值
        hs = torch.div(torch.sum(seq_h, 1), session_len)
        # 扩展平均值张量，使其维度与 seq_h 匹配
        hs = hs.unsqueeze(-2).repeat(1, len, 1)
        # 扩展最后一个物品的嵌入向量，使其维度与 seq_h 匹配
        hl = hl.unsqueeze(-2).repeat(1, len, 1)
        # 将位置嵌入向量和物品序列嵌入向量拼接，并通过线性层进行变换
        nh = self.w_1(torch.cat([pos_emb, seq_h], -1))
        # 通过 GLU 门控机制计算 nh
        nh = torch.sigmoid(self.glu1(nh) + self.glu2(hs) + self.glu3(hl))
        # 通过线性层计算 beta
        beta = self.w_2(nh)
        # 应用掩码到 beta 上
        beta = beta * mask
        # 计算兴趣偏好
        interest_pre = torch.sum(beta * seq_h, 1)

        # 多任务学习部分
        # 通过 tanh 激活函数计算 m_p_i
        m_p_i = torch.tanh(self.merge_w1(interest_pre) + self.merge_w2(price_pre))
        # 通过 sigmoid 激活函数计算价格偏好的门控值
        g_p = torch.sigmoid(self.merge_w3(price_pre) + self.merge_w4(m_p_i))
        # 通过 sigmoid 激活函数计算兴趣偏好的门控值
        g_i = torch.sigmoid(self.merge_w5(interest_pre) + self.merge_w6(m_p_i))

        # 融合价格偏好和兴趣偏好
        p_pre = g_p * price_pre + (1 - g_p) * interest_pre
        # 融合兴趣偏好和价格偏好
        i_pre = g_i * interest_pre + (1 - g_i) * price_pre

        return i_pre, p_pre

    def transpose_for_scores(self, x, attention_head_size):
        """
        调整输入张量的形状，以便进行多头注意力计算。

        参数:
        x (torch.Tensor): 输入的张量，形状为 [bs, seqlen, hid_size]，其中 bs 是批量大小，seqlen 是序列长度，hid_size 是隐藏层维度。
        attention_head_size (int): 每个注意力头的维度。

        返回:
        torch.Tensor: 调整形状后的张量，形状为 [bs, num_heads, seqlen, attention_head_size]。
        """
        # INPUT:  x'shape = [bs, seqlen, hid_size]
        # 计算新的张量形状，将隐藏层维度拆分为多个注意力头
        new_x_shape = x.size()[:-1] + (self.num_heads, attention_head_size)  # [bs, seqlen, 8, 16]
        # 调整张量的形状
        x = x.view(*new_x_shape)  
        # 重新排列张量的维度，以便进行多头注意力计算
        return x.permute(0, 2, 1, 3)


    def forward(self, session_item, price_seqs, session_len, reversed_sess_item, mask):
        """
        模型的前向传播方法，计算物品嵌入、价格嵌入、会话嵌入和价格偏好等信息。

        参数:
        session_item (torch.Tensor): 一个批次内的所有会话序列，形状为 [batch_size, max_seq_len]。
        price_seqs (torch.Tensor): 会话对应的价格序列，形状为 [batch_size, max_seq_len]。
        session_len (torch.Tensor): 每个会话的实际长度，形状为 [batch_size]。
        reversed_sess_item (torch.Tensor): 反转后的会话物品序列，形状为 [batch_size, max_seq_len]。
        mask (torch.Tensor): 掩码张量，用于屏蔽无效位置，形状为 [batch_size, max_seq_len]。

        返回:
        torch.Tensor: 经过超图卷积更新后的物品嵌入向量。
        torch.Tensor: 经过超图卷积更新后的价格嵌入向量。
        torch.Tensor: 批次内会话的嵌入向量。
        torch.Tensor: 批次内会话的价格偏好向量。
        torch.Tensor: 物品对应的价格嵌入向量。
        torch.Tensor: 物品对应的价格索引。
        """
        # session_item 是一个batch里的所有session [[23,34,0,0],[1,3,4,0]]
        # 调用 HyperGraph 模块，经过三次 GCN 迭代得到所有物品和价格的嵌入向量
        item_embeddings_hg, price_embeddings_hg = self.HyperGraph(self.adjacency, self.adjacency_pp, self.adjacency_cc, self.adjacency_bb, self.adjacency_vp, self.adjacency_vc, self.adjacency_vb, self.adjacency_pv, self.adjacency_pc, self.adjacency_pb, self.adjacency_cv, self.adjacency_cp, self.adjacency_cb, self.adjacency_bv, self.adjacency_bp, self.adjacency_bc, self.embedding.weight, self.price_embedding.weight, self.category_embedding.weight, self.brand_embedding.weight) 
        # 调用 generate_sess_emb 方法，计算批次内会话的嵌入向量和价格偏好向量
        sess_emb_hgnn, sess_pri_hgnn = self.generate_sess_emb(item_embeddings_hg, price_embeddings_hg, session_item, price_seqs, session_len, reversed_sess_item, mask) 
        # 获取物品到价格邻接矩阵的行索引
        v_table = self.adjacency_vp.row
        # 对行索引进行排序，得到排序后的索引
        temp, idx = torch.sort(torch.tensor(v_table), dim=0, descending=False)
        # 根据排序后的索引获取对应的列索引
        vp_idx = self.adjacency_vp.col[idx]
        # 根据列索引获取对应的价格嵌入向量
        item_pri_l = price_embeddings_hg[vp_idx]

        return item_embeddings_hg, price_embeddings_hg, sess_emb_hgnn, sess_pri_hgnn, item_pri_l, vp_idx



def predict(model, i, data):
    """
    根据模型、批次索引和数据集预测兴趣目标、价格目标以及对应的分数。

    参数:
    model (torch.nn.Module): 训练好的模型实例。
    i (int): 批次索引，用于从数据集中获取特定批次的数据。
    DBLP (object): 数据集对象，包含数据切片和重叠信息等方法。

    返回:
    torch.Tensor: 兴趣目标的张量。
    torch.Tensor: 价格目标的张量。
    torch.Tensor: 兴趣分数的张量。
    torch.Tensor: 价格分数的张量。
    """
    # 从数据集中获取一个批次的数据，包括目标、会话长度、会话物品序列、反转的会话物品序列、掩码和价格序列
    tar, session_len, session_item, reversed_sess_item, mask, price_seqs = data.get_slice(i) 
    # A_hat, D_hat = DBLP.get_overlap(session_item)
    # 将数据转换为 torch.Tensor 类型，并移动到 GPU 上（如果可用）
    session_item = trans_to_cuda(torch.Tensor(session_item).long())
    session_len = trans_to_cuda(torch.Tensor(session_len).long())
    price_seqs = trans_to_cuda(torch.Tensor(price_seqs).long())
    # A_hat = trans_to_cuda(torch.Tensor(A_hat))
    # D_hat = trans_to_cuda(torch.Tensor(D_hat))
    tar_interest = trans_to_cuda(torch.Tensor(tar).long())
    mask = trans_to_cuda(torch.Tensor(mask).long())
    reversed_sess_item = trans_to_cuda(torch.Tensor(reversed_sess_item).long())
    # 通过模型前向传播得到物品嵌入、价格嵌入、会话嵌入、会话价格偏好、物品价格嵌入和物品价格索引
    item_emb_hg, price_emb_hg, sess_emb_hgnn, sess_pri_hgnn, item_pri_l, vp_idx = model(session_item, price_seqs, session_len, reversed_sess_item, mask)
    # 计算会话嵌入与物品嵌入的矩阵乘法，得到兴趣分数
    scores_interest = torch.mm(sess_emb_hgnn, torch.transpose(item_emb_hg, 1, 0))
    # 计算会话价格偏好与价格嵌入的矩阵乘法，得到价格分数
    scores_price = torch.mm(sess_pri_hgnn, torch.transpose(price_emb_hg, 1, 0))
    # 获取目标对应的价格索引，并转换为 torch.Tensor 类型，移动到 GPU 上
    tar_price = trans_to_cuda(torch.Tensor(vp_idx[tar]).long())
    return tar_interest, tar_price, scores_interest, scores_price


def infer(model, i, data):
    """
    根据模型、批次索引和数据集进行推理，得到目标和综合分数。

    参数:
    model (torch.nn.Module): 训练好的模型实例。
    i (int): 批次索引，用于从数据集中获取特定批次的数据。
    DBLP (object): 数据集对象，包含数据切片和重叠信息等方法。

    返回:
    torch.Tensor: 目标的张量。
    torch.Tensor: 综合分数的张量。
    """
    # 从数据集中获取一个批次的数据，包括目标、会话长度、会话物品序列、反转的会话物品序列、掩码和价格序列
    tar, session_len, session_item, reversed_sess_item, mask, price_seqs = data.get_slice(i) 
    # A_hat, D_hat = DBLP.get_overlap(session_item)
    # 将数据转换为 torch.Tensor 类型，并移动到 GPU 上（如果可用）
    session_item = trans_to_cuda(torch.Tensor(session_item).long())
    session_len = trans_to_cuda(torch.Tensor(session_len).long())
    price_seqs = trans_to_cuda(torch.Tensor(price_seqs).long())
    # A_hat = trans_to_cuda(torch.Tensor(A_hat))
    # D_hat = trans_to_cuda(torch.Tensor(D_hat))
    tar = trans_to_cuda(torch.Tensor(tar).long())
    mask = trans_to_cuda(torch.Tensor(mask).long())
    reversed_sess_item = trans_to_cuda(torch.Tensor(reversed_sess_item).long())
    # 通过模型前向传播得到物品嵌入、价格嵌入、会话嵌入、会话价格偏好、物品价格嵌入和物品价格索引
    item_emb_hg, price_emb_hg, sess_emb_hgnn, sess_pri_hgnn, item_pri_l, vp_idx = model(session_item, price_seqs, session_len, reversed_sess_item, mask)
    # 计算会话嵌入与物品嵌入的矩阵乘法，得到兴趣分数
    scores_interest = torch.mm(sess_emb_hgnn, torch.transpose(item_emb_hg, 1, 0))
    # 计算会话价格偏好与物品价格嵌入的矩阵乘法，得到价格分数
    scores_price = torch.mm(sess_pri_hgnn, torch.transpose(item_pri_l, 1, 0))
    # 对兴趣分数和价格分数分别应用 Softmax 函数，并将结果相加得到综合分数
    # scores = (1-model.beta) * torch.softmax(scores_interest, 1) + model.beta * torch.softmax(scores_price, 1)
    scores = torch.softmax(scores_interest, 1) + torch.softmax(scores_price, 1)
    return tar, scores


def train_test(model, train_data, test_data):
    """
    训练模型并在测试集上进行评估，返回评估指标和总损失。

    参数:
    model (torch.nn.Module): 待训练和评估的模型实例。
    train_data (object): 训练数据集对象，包含数据切片和生成批次等方法。
    test_data (object): 测试数据集对象，包含数据切片和生成批次等方法。

    返回:
    dict: 包含不同 K 值下的命中率、平均倒数排名和归一化折损累积增益的评估指标字典。
    torch.Tensor: 训练过程中的总损失。
    """
    # 打印训练开始时间
    print('start training: ', datetime.datetime.now())
    # 开启自动求导异常检测
    torch.autograd.set_detect_anomaly(True)
    # 初始化总损失为 0
    total_loss = 0.0
    # 从训练数据集中生成批次索引
    slices = train_data.generate_batch(model.batch_size) 
    # 遍历每个批次
    for i in slices:
        # 清空模型的梯度
        model.zero_grad()
        # 调用 predict 函数得到兴趣目标、价格目标、兴趣分数和价格分数
        tar_interest, tar_price, scores_interest, scores_price = predict(model, i, train_data)
        # 计算兴趣分数的损失
        loss_interest = model.loss_function(scores_interest + 1e-8, tar_interest)
        # 计算价格分数的损失
        loss_price = model.loss_function(scores_price + 1e-8, tar_price)
        # 总损失为兴趣损失和价格损失之和
        loss = loss_interest + loss_price
        # 反向传播计算梯度
        loss.backward()
        #        print(loss.item())
        # 更新模型参数
        model.optimizer.step()
        # 累加总损失
        total_loss += loss
    # 打印训练总损失
    print('\tLoss:\t%.3f' % total_loss)
    # 定义不同的 K 值，用于评估指标计算
    top_K = [1, 5, 10, 20]
    # 初始化评估指标字典
    metrics = {}
    # 为每个 K 值初始化命中率、平均倒数排名和归一化折损累积增益的列表
    for K in top_K:
        metrics['hit%d' % K] = []
        metrics['mrr%d' % K] = []
        metrics['ndcg%d' % K] = []
    # 打印预测开始时间
    print('start predicting: ', datetime.datetime.now())
    # 将模型设置为评估模式
    model.eval()
    # 从测试数据集中生成批次索引
    slices = test_data.generate_batch(model.batch_size)
    # 遍历每个批次
    for i in slices:
        # 调用 infer 函数得到目标和综合分数
        tar, scores = infer(model, i, test_data)
        # 将分数转换为 CPU 上的 numpy 数组
        scores = trans_to_cpu(scores).detach().numpy()
        # 对分数进行排序，得到预测结果的索引
        index = np.argsort(-scores, 1)
        # 将目标转换为 CPU 上的 numpy 数组
        tar = trans_to_cpu(tar).detach().numpy()
        # 计算不同 K 值下的评估指标
        for K in top_K:
            for prediction, target in zip(index[:, :K], tar):
                # 计算命中率
                metrics['hit%d' % K].append(np.isin(target, prediction))
                if len(np.where(prediction == target)[0]) == 0:
                    # 如果预测结果中不包含目标，则平均倒数排名和归一化折损累积增益为 0
                    metrics['mrr%d' % K].append(0)
                    metrics['ndcg%d' % K].append(0)
                else:
                    # 计算平均倒数排名和归一化折损累积增益
                    metrics['mrr%d' % K].append(1 / (np.where(prediction == target)[0][0] + 1))
                    metrics['ndcg%d' % K].append(1 / (np.log2(np.where(prediction == target)[0][0] + 2)))
    return metrics, total_loss


