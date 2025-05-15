# 导入 argparse 模块，用于解析命令行参数
import argparse
# 导入 pickle 模块，用于序列化和反序列化 Python 对象
import pickle
# 导入 time 模块，用于处理时间相关操作
import time
# 从 util 模块中导入 Data 和 split_validation 类/函数
from util import Data, split_validation
# 从 model 模块中导入所有内容
from model import *
# 导入 os 模块，用于与操作系统进行交互
import os

# 创建一个 ArgumentParser 对象，用于解析命令行参数
parser = argparse.ArgumentParser()
# 添加 --dataset 参数，指定数据集名称，默认值为 'Grocery_and_Gourmet_Food'
parser.add_argument('--dataset', default='Grocery_and_Gourmet_Food', help='dataset name: 2019-Oct/Grocery_and_Gourmet_Food/Toys_and_Games')
# 添加 --epoch 参数，指定训练的轮数，类型为整数，默认值为 20
parser.add_argument('--epoch', type=int, default=20, help='number of epochs to train for')
# 添加 --batchSize 参数，指定输入批次的大小，类型为整数，默认值为 100
parser.add_argument('--batchSize', type=int, default=100, help='input batch size')
# 添加 --embSize 参数，指定嵌入向量的大小，类型为整数，默认值为 128
parser.add_argument('--embSize', type=int, default=128, help='embedding size')
# 添加 --num_heads 参数，指定注意力头的数量，类型为整数，默认值为 8
parser.add_argument('--num_heads', type=int, default=8, help='number of attention heads')
# 添加 --l2 参数，指定 L2 正则化的惩罚系数，类型为浮点数，默认值为 1e-5
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')
# 添加 --lr 参数，指定学习率，类型为浮点数，默认值为 0.001
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
# 添加 --layer 参数，指定使用的层数，类型为浮点数，默认值为 2
parser.add_argument('--layer', type=float, default=2, help='the number of layer used')
# 添加 --beta 参数，指定价格任务的幅度，类型为浮点数，默认值为 0.2
parser.add_argument('--beta', type=float, default=0.2, help='price task maginitude')
# 添加 --filter 参数，指定是否过滤关联矩阵，类型为布尔值，默认值为 False
parser.add_argument('--filter', type=bool, default=False, help='filter incidence matrix')

# 解析命令行参数并将结果存储在 opt 变量中
opt = parser.parse_args()
# 打印解析后的命令行参数
print(opt)
# 注释掉的代码，用于设置可见的 CUDA 设备
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
# 设置当前使用的 CUDA 设备为 1
torch.cuda.set_device(1)

def main():
    # 从文件中加载训练数据，文件路径根据命令行参数指定的数据集名称生成
    # list[0]:session list[1]:label
    train_data = pickle.load(open('./datasets/' + opt.dataset + '/train.txt', 'rb'))
    # 从文件中加载测试数据，文件路径根据命令行参数指定的数据集名称生成
    test_data = pickle.load(open('./datasets/' + opt.dataset + '/test.txt', 'rb'))

    # 根据不同的数据集名称，设置节点数、价格数、类别数和品牌数
    if opt.dataset == 'Grocery_and_Gourmet_Food':
        n_node = 6230
        n_price = 5
        n_category = 550
        n_brand = 1306
    elif opt.dataset == 'Toys_and_Games':
        n_node = 18979
        n_price = 5
        n_category = 430
        n_brand = 1429
    elif opt.dataset == '2019-Oct':
        n_node = 13026
        n_price = 10
        n_category = 226
        n_brand = 148
    else:
        # 如果数据集名称未知，打印提示信息
        print("unkonwn dataset")
    # 数据格式: sessions, price_seq, matrix_session_item, matrix_session_price, matrix_pv, matrix_pb, matrix_pc, matrix_bv, matrix_bc, matrix_cv
    # 对训练数据进行处理，设置打乱顺序并传入节点数、价格数、类别数和品牌数
    train_data = Data(train_data, shuffle=True, n_node=n_node, n_price=n_price, n_category=n_category, n_brand=n_brand)
    # 对测试数据进行处理，设置打乱顺序并传入节点数、价格数、类别数和品牌数
    test_data = Data(test_data, shuffle=True, n_node=n_node, n_price=n_price, n_category=n_category, n_brand=n_brand)
    # 将模型移动到 CUDA 设备上
    """p:price价格，c:category类别，b:brand品牌，v:物品，p:purchase购买"""
    model = trans_to_cuda(
        # 初始化 DHCN 模型，传入训练数据的邻接矩阵、节点数、价格数、类别数、品牌数等参数 3+2*6=15 + ss
        DHCN(adjacency=train_data.adjacency, 
             adjacency_pp=train_data.adjacency_pp, adjacency_cc=train_data.adjacency_cc,
             adjacency_bb=train_data.adjacency_bb, adjacency_vp=train_data.adjacency_vp,
             adjacency_vc=train_data.adjacency_vc, adjacency_vb=train_data.adjacency_vb,
             adjacency_pv=train_data.adjacency_pv, adjacency_pc=train_data.adjacency_pc,
             adjacency_pb=train_data.adjacency_pb, adjacency_cv=train_data.adjacency_cv,
             adjacency_cp=train_data.adjacency_cp, adjacency_cb=train_data.adjacency_cb,
             adjacency_bv=train_data.adjacency_bv, adjacency_bp=train_data.adjacency_bp,
             adjacency_bc=train_data.adjacency_bc, 
             n_node=n_node, n_price=n_price, n_category=n_category,
             n_brand=n_brand, lr=opt.lr, layers=opt.layer, l2=opt.l2, beta=opt.beta, dataset=opt.dataset,
             num_heads=opt.num_heads, emb_size=opt.embSize, batch_size=opt.batchSize))

    # 定义评估指标的 top-K 值列表
    top_K = [1, 5, 10, 20]
    # 初始化最佳结果字典
    best_results = {}
    # 为每个 top-K 值初始化最佳结果和对应的轮数
    for K in top_K:
        best_results['epoch%d' % K] = [0, 0, 0]
        best_results['metric%d' % K] = [0, 0, 0]

    # 开始训练循环，训练轮数由命令行参数指定
    for epoch in range(opt.epoch):
        print('-------------------------------------------------------')
        print('epoch: ', epoch)
        # 调用 train_test 函数进行训练和测试，返回评估指标和总损失
        metrics, total_loss = train_test(model, train_data, test_data)
        # 对每个 top-K 值的评估指标进行处理，计算平均值并乘以 100
        for K in top_K:
            metrics['hit%d' % K] = np.mean(metrics['hit%d' % K]) * 100
            metrics['mrr%d' % K] = np.mean(metrics['mrr%d' % K]) * 100
            metrics['ndcg%d' % K] = np.mean(metrics['ndcg%d' % K]) * 100
            # 如果当前轮数的命中率高于之前的最佳值，更新最佳值和对应的轮数
            if best_results['metric%d' % K][0] < metrics['hit%d' % K]:
                best_results['metric%d' % K][0] = metrics['hit%d' % K]
                best_results['epoch%d' % K][0] = epoch
            # 如果当前轮数的 MRR 高于之前的最佳值，更新最佳值和对应的轮数
            if best_results['metric%d' % K][1] < metrics['mrr%d' % K]:
                best_results['metric%d' % K][1] = metrics['mrr%d' % K]
                best_results['epoch%d' % K][1] = epoch
            # 如果当前轮数的 NDCG 高于之前的最佳值，更新最佳值和对应的轮数
            if best_results['metric%d' % K][2] < metrics['ndcg%d' % K]:
                best_results['metric%d' % K][2] = metrics['ndcg%d' % K]
                best_results['epoch%d' % K][2] = epoch
        # 打印当前轮数的评估指标
        print(metrics)
        # 注释掉的代码，用于打印训练损失和最佳评估指标及对应的轮数
        # for K in top_K:
        #     print('train_loss:\t%.4f\tRecall@%d: %.4f\tMRR%d: %.4f\tNDCG%d: %.4f\tEpoch: %d,  %d, %d' %
        #           (total_loss, K, best_results['metric%d' % K][0], K, best_results['metric%d' % K][1],K, best_results['metric%d' % K][2],
        #            best_results['epoch%d' % K][0], best_results['epoch%d' % K][1], best_results['epoch%d' % K][2]))
        # 打印评估指标的表头
        print('P@1\tP@5\tM@5\tN@5\tP@10\tM@10\tN@10\tP@20\tM@20\tN@20\t')
        # 打印最佳评估指标
        print("%.2f\t %.2f\t %.2f\t %.2f\t %.2f\t %.2f\t %.2f\t %.2f\t %.2f\t %.2f" % (
            best_results['metric1'][0], best_results['metric5'][0], best_results['metric5'][1],
            best_results['metric5'][2], best_results['metric10'][0], best_results['metric10'][1],
            best_results['metric10'][2], best_results['metric20'][0], best_results['metric20'][1],
            best_results['metric20'][2]))
        # 打印最佳评估指标对应的轮数
        print("%d\t %d\t %d\t %d\t %d\t %d\t %d\t %d\t %d\t %d" % (
            best_results['epoch1'][0], best_results['epoch5'][0], best_results['epoch5'][1],
            best_results['epoch5'][2], best_results['epoch10'][0], best_results['epoch10'][1],
            best_results['epoch10'][2], best_results['epoch20'][0], best_results['epoch20'][1],
            best_results['epoch20'][2]))

if __name__ == '__main__':
    # 当脚本作为主程序运行时，调用 main 函数
    main()
