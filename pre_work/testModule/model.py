import torch
import torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score
from torch import optim
from torch_geometric.nn import HypergraphConv
class HHGNN(nn.Module):
    def __init__(self,in_dims,hyperedge_index,node_types):
        self.hyperedge_index =hyperedge_index
        self.num_edges =torch.max(self.hyperedge_index[0])+1
        self.feature_dim=3
        self.mid_dim=3
        self.hidden_dim=3
        self.output_dim=2
        self.edge_dim=3*self.mid_dim
        self.classify_dim=13
        super().__init__()
        self.projections=nn.ModuleDict({
            t: nn.Linear(in_dims[t], self.feature_dim) for t in node_types
        })

        self.conv1 = HypergraphConv(self.feature_dim, self.hidden_dim)
        self.conv_relu1 = nn.ReLU()
        self.conv2 = HypergraphConv(self.hidden_dim, self.mid_dim)
        self.conv_relu2 = nn.ReLU()

        self.classify_layer1 =nn.Linear(self.edge_dim, self.classify_dim)
        self.classify_relu1 = nn.ReLU()
        self.classify_layer2 = nn.Linear(self.classify_dim, self.classify_dim)
        self.classify_relu2 = nn.ReLU()
        self.classify_layer3 = nn.Linear(self.classify_dim,self.output_dim)



    def forward(self,x_dict):
        z=[]
        for t in x_dict: #x_dict:类型，tensor
            z.append(self.projections[t](x_dict[t]))
        x=torch.cat(z,dim=0)
        x=self.conv1(x, self.hyperedge_index)
        x=self.conv_relu1(x)
        x=self.conv2(x, self.hyperedge_index)
        x=self.conv_relu2(x)
        x=self.get_nodes_of_edge(x)
        x=self.classify_layer1(x)
        x=self.classify_relu1(x)
        x=self.classify_layer2(x)
        x=self.classify_relu2(x)
        x=self.classify_layer3(x)
        return x

    def get_nodes_of_edge(self, x):
        num_edges = torch.max(self.hyperedge_index[1]) + 1
        edge_features_list = []

        for edge_idx in range(num_edges):
            # 获取连接当前超边的节点索引
            node_indices = self.hyperedge_index[0][self.hyperedge_index[1] == edge_idx]
            # 确保每条超边连接3个节点
            assert len(node_indices) == 3, f"超边 {edge_idx} 没有连接3个节点。"

            # 获取这三个节点的特征
            node_features = x[node_indices]
            # 拼接节点特征
            edge_feature = node_features.flatten()
            edge_features_list.append(edge_feature)
            # print(f"edge_feature:{edge_feature}")
        # 将所有超边的特征拼接成一个张量
        edge_features = torch.stack(edge_features_list)
        # print(f"edge_features:{edge_features}")
        return edge_features


    def train_(self, model, optimizer, criterion,edge_attr,x):
        model.train()
        optimizer.zero_grad()

        out = model(x)
        loss = criterion(out, edge_attr)

        loss.backward()
        optimizer.step()

        return loss.item()
    def test(self, model, x,edge_attr):
        model.eval()
        with torch.no_grad():
            out = model(x)
            # print(f"out:{out}")
            # 获取预测类别
            _, pred = torch.max(out, 1)
            # print(f"edge_attr:{edge_attr}")
            acc = accuracy_score(edge_attr.cpu(), pred.cpu())
            f1 = f1_score(edge_attr.cpu(), pred.cpu(), average='macro')
        return acc, f1
# class MLP(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.input_dim=3
#         self.hidden_dim=12
#         self.output_dim=2
#         self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
#         self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
#         self.fc3 = nn.Linear(self.hidden_dim, self.output_dim)
#         self.relu = nn.ReLU()
#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.relu(x)
#         x = self.fc2(x)
#         x = self.relu(x)
#         x = self.fc3(x)
#         return x
#
#     def compute_accuracy(s,outputs, targets):
#         _, predicted = torch.max(outputs, 1)  # 获取预测类别（最大值的索引）
#         correct = (predicted == targets).sum().item()
#         accuracy = correct / targets.size(0)
#         return accuracy
# def main():
#     # 设置随机种子（保证可重复性）
#     torch.manual_seed(42)
#
#     # 创建模型
#     model = MLP()
#
#     # 定义优化器和损失函数
#     optimizer = optim.Adam(model.parameters(), lr=0.001)
#     criterion = nn.CrossEntropyLoss()  # 分类任务
#
#     n=100
#     # 生成随机输入（5个样本，3维特征）
#     inputs = torch.randn(n, 3)
#
#     # 生成随机目标（5个样本，2维输出）
#     targets = torch.randint(0, 2, (n,))  # 5个样本，类别0或1（二分类）
#
#     # 训练循环（示例：1个epoch）
#     for epoch in range(100000):
#         optimizer.zero_grad()  # 清空梯度
#         outputs = model(inputs)  # 前向传播
#         loss = criterion(outputs, targets)  # 计算损失
#         loss.backward()  # 反向传播
#         optimizer.step()  # 更新权重
#         accuracy = model.compute_accuracy(outputs, targets)# 计算准确率
#         print(f"Loss: {loss.item():.4f}, Accuracy: {accuracy * 100:.2f}%")
#
#
# if __name__ == "__main__":
#     main()