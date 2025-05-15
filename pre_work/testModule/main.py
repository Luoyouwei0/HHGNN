from model import *
from data import *
import torch
import torch.nn as nn
def main():
    epochs=10000
    node_types,in_dims,hyperedge_index,edge_attr,x_dict=make_data()
    model = HHGNN(in_dims,hyperedge_index,node_types)


    # 优化器和损失函数
    optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # 训练循环
    for epoch in range(1, epochs + 1):
        loss = model.train_(model, optimizer, criterion,edge_attr,x_dict)
        train_acc, train_f1 = model.test(model, x_dict,edge_attr)
        test_acc, test_f1 = model.test(model, x_dict,edge_attr)
        if True:
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, '
                f'Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}, '
                f'Train F1: {train_f1:.4f}, Test F1: {test_f1:.4f}')


if __name__ == '__main__':
    main()