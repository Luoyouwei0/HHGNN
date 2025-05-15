# 导入 PyTorch 库，用于深度学习任务
import torch
# 从 sklearn.metrics 模块导入 f1_score 函数，用于计算 F1 分数
from sklearn.metrics import f1_score
# 从 utils 模块导入 EarlyStopping 类和 load_data 函数
from utils import EarlyStopping, load_data


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
def evaluate(model, g, features, labels, mask, loss_func):
    # 将模型设置为评估模式，关闭 Dropout 等训练时使用的特殊层
    model.eval()
    # 上下文管理器，不计算梯度，提高计算效率
    with torch.no_grad():
        # 前向传播，得到模型的输出
        logits = model(g, features)
    # 计算损失值
    loss = loss_func(logits[mask], labels[mask])
    # 调用 score 函数，计算准确率、微观 F1 分数和宏观 F1 分数
    accuracy, micro_f1, macro_f1 = score(logits[mask], labels[mask])

    # 返回损失值、准确率、微观 F1 分数和宏观 F1 分数
    return loss, accuracy, micro_f1, macro_f1


# 定义主函数，包含模型训练和评估的主要逻辑
def main(args):
    # 如果 args['hetero'] 为 True，则 g 是一个异构图；否则，它是一个同构图列表
    # 调用 load_data 函数，加载数据集并返回相关数据
    (
        g,
        features,
        labels,
        num_classes,
        train_idx,
        val_idx,
        test_idx,
        train_mask,
        val_mask,
        test_mask,
    ) = load_data(args["dataset"])

    # 检查 PyTorch 是否有 BoolTensor 类型
    if hasattr(torch, "BoolTensor"):
        # 将训练集、验证集和测试集的掩码转换为布尔类型
        train_mask = train_mask.bool()
        val_mask = val_mask.bool()
        test_mask = test_mask.bool()

    # 将特征张量移动到指定设备（如 GPU）上
    features = features.to(args["device"])
    # 将标签张量移动到指定设备上
    labels = labels.to(args["device"])
    # 将训练集掩码张量移动到指定设备上
    train_mask = train_mask.to(args["device"])
    # 将验证集掩码张量移动到指定设备上
    val_mask = val_mask.to(args["device"])
    # 将测试集掩码张量移动到指定设备上
    test_mask = test_mask.to(args["device"])

    # 如果使用异构图
    if args["hetero"]:
        # 从 model_hetero 模块导入 HAN 模型
        from model_hetero import HAN

        # 初始化异构图 HAN 模型
        model = HAN(
            meta_paths=[["pa", "ap"], ["pf", "fp"]],
            in_size=features.shape[1],
            hidden_size=args["hidden_units"],
            out_size=num_classes,
            num_heads=args["num_heads"],
            dropout=args["dropout"],
        ).to(args["device"])
        # 将异构图移动到指定设备上
        g = g.to(args["device"])
    else:
        # 从 model 模块导入 HAN 模型
        from model import HAN

        # 初始化同构图 HAN 模型
        model = HAN(
            num_meta_paths=len(g),
            in_size=features.shape[1],
            hidden_size=args["hidden_units"],
            out_size=num_classes,
            num_heads=args["num_heads"],
            dropout=args["dropout"],
        ).to(args["device"])
        # 将同构图列表中的每个图移动到指定设备上
        g = [graph.to(args["device"]) for graph in g]

    # 初始化早停机制，设置耐心值
    stopper = EarlyStopping(patience=args["patience"])
    # 定义损失函数，使用交叉熵损失
    loss_fcn = torch.nn.CrossEntropyLoss()
    # 定义优化器，使用 Adam 优化器
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args["lr"], weight_decay=args["weight_decay"]
    )

    # 开始训练循环，迭代指定的轮数
    for epoch in range(args["num_epochs"]):
        # 将模型设置为训练模式，开启 Dropout 等训练时使用的特殊层
        model.train()
        # 前向传播，得到模型的输出
        logits = model(g, features)
        # 计算训练损失
        loss = loss_fcn(logits[train_mask], labels[train_mask])

        # 梯度清零，避免梯度累积
        optimizer.zero_grad()
        # 反向传播，计算梯度
        loss.backward()
        # 更新模型参数
        optimizer.step()

        # 计算训练集的准确率、微观 F1 分数和宏观 F1 分数
        train_acc, train_micro_f1, train_macro_f1 = score(
            logits[train_mask], labels[train_mask]
        )
        # 调用 evaluate 函数，计算验证集的损失值、准确率、微观 F1 分数和宏观 F1 分数
        val_loss, val_acc, val_micro_f1, val_macro_f1 = evaluate(
            model, g, features, labels, val_mask, loss_fcn
        )
        # 调用早停机制的 step 方法，判断是否需要提前停止训练
        early_stop = stopper.step(val_loss.data.item(), val_acc, model)

        # 打印训练和验证的指标
        print(
            "Epoch {:d} | Train Loss {:.4f} | Train Micro f1 {:.4f} | Train Macro f1 {:.4f} | "
            "Val Loss {:.4f} | Val Micro f1 {:.4f} | Val Macro f1 {:.4f}".format(
                epoch + 1,
                loss.item(),
                train_micro_f1,
                train_macro_f1,
                val_loss.item(),
                val_micro_f1,
                val_macro_f1,
            )
        )

        # 如果满足早停条件，跳出训练循环
        if early_stop:
            break

    # 加载早停机制保存的最佳模型参数
    stopper.load_checkpoint(model)
    # 调用 evaluate 函数，计算测试集的损失值、准确率、微观 F1 分数和宏观 F1 分数
    test_loss, test_acc, test_micro_f1, test_macro_f1 = evaluate(
        model, g, features, labels, test_mask, loss_fcn
    )
    # 打印测试集的指标
    print(
        "Test loss {:.4f} | Test Micro f1 {:.4f} | Test Macro f1 {:.4f}".format(
            test_loss.item(), test_micro_f1, test_macro_f1
        )
    )


if __name__ == "__main__":
    # 导入 argparse 模块，用于解析命令行参数
    import argparse

    # 从 utils 模块导入 setup 函数
    from utils import setup

    # 创建一个 ArgumentParser 对象，用于解析命令行参数
    parser = argparse.ArgumentParser("HAN")
    # 添加一个名为 -s 或 --seed 的参数，用于设置随机种子，默认值为 1
    parser.add_argument("-s", "--seed", type=int, default=1, help="Random seed")
    # 添加一个名为 -ld 或 --log-dir 的参数，用于指定保存训练结果的目录，默认值为 "results"
    parser.add_argument(
        "-ld",
        "--log-dir",
        type=str,
        default="results",
        help="Dir for saving training results",
    )
    # 添加一个名为 --hetero 的参数，用于指定是否使用异构图，默认值为 False
    parser.add_argument(
        "--hetero",
        action="store_true",
        help="Use metapath coalescing with DGL's own dataset",
    )
    # 解析命令行参数，并将结果转换为字典
    args = parser.parse_args().__dict__

    # 调用 setup 函数，对参数进行进一步设置
    args = setup(args)

    # 调用 main 函数，开始训练和评估模型
    main(args)
