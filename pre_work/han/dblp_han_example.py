import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.data import HGBDataset
from dgl.nn.pytorch import HANLayer
from sklearn.metrics import f1_score

# 设置随机种子（可选）
torch.manual_seed(42)

# 1. 加载DBLP数据集（使用DGL的HGBDataset）
dataset = HGBDataset(name='dblp4area', raw_dir='./DBLP')
graph = dataset[0]  # 获取异质图

# 打印图信息（确认节点和边类型）
print("Node types:", graph.ntypes)
print("Edge types:", graph.etypes)


# 2. 定义HAN模型
class HAN(nn.Module):
    def __init__(self, num_metapaths, in_size, hidden_size, out_size, num_heads, dropout):
        super(HAN, self).__init__()
        # HAN层（支持多注意力头）
        self.layers = nn.ModuleList()
        self.layers.append(HANLayer(num_metapaths, in_size, hidden_size, num_heads[0], dropout))
        self.layers.append(HANLayer(num_metapaths, hidden_size * num_heads[0], out_size, num_heads[1], dropout))

    def forward(self, g, h_dict):
        # 逐层传播
        for layer in self.layers:
            h_dict = layer(g, h_dict)
        return h_dict


# 3. 数据预处理
# 假设DBLP的节点类型为 'author', 'paper', 'conference', 'term'
# 目标：对作者节点（'author'）进行分类

# 获取作者节点的特征和标签
author_feats = graph.nodes['author'].data['feat']
author_labels = graph.nodes['author'].data['label']

# 划分训练/验证/测试集（示例：60%/20%/20%）
num_authors = graph.num_nodes('author')
train_mask = torch.zeros(num_authors, dtype=torch.bool)
val_mask = torch.zeros(num_authors, dtype=torch.bool)
test_mask = torch.zeros(num_authors, dtype=torch.bool)

train_mask[:int(0.6 * num_authors)] = True
val_mask[int(0.6 * num_authors):int(0.8 * num_authors)] = True
test_mask[int(0.8 * num_authors):] = True

# 4. 定义元路径（关键步骤！）
# 根据DBLP的边类型定义元路径（示例：APA表示作者-论文-作者）
metapaths = {
    'APA': [('author', 'author-paper', 'paper'), ('paper', 'paper-author', 'author')],
    'APCPA': [('author', 'author-paper', 'paper'), ('paper', 'paper-conference', 'conference'),
              ('conference', 'conference-paper', 'paper'), ('paper', 'paper-author', 'author')]
}

# 将元路径转换为DGL格式的邻接矩阵列表
metapath_graphs = []
for mp_name, mp_edges in metapaths.items():
    metapath_graphs.append(dgl.metapath_reachable_graph(graph, mp_edges))

# 5. 初始化模型和优化器
in_size = author_feats.shape[1]
hidden_size = 64
out_size = torch.max(author_labels).item() + 1  # 类别数
num_heads = [4, 1]  # 每层的注意力头数
dropout = 0.5

model = HAN(num_metapaths=len(metapaths),
            in_size=in_size,
            hidden_size=hidden_size,
            out_size=out_size,
            num_heads=num_heads,
            dropout=dropout)

optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.001)


# 6. 训练函数
def train():
    model.train()
    logits = model(metapath_graphs, {'author': author_feats})['author']
    loss = F.cross_entropy(logits[train_mask], author_labels[train_mask])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


# 7. 评估函数
def evaluate(mask):
    model.eval()
    with torch.no_grad():
        logits = model(metapath_graphs, {'author': author_feats})['author']
        preds = torch.argmax(logits[mask], dim=1)
        labels = author_labels[mask]
        f1 = f1_score(labels.cpu(), preds.cpu(), average='macro')
    return f1


# 8. 训练循环
epochs = 100
for epoch in range(epochs):
    loss = train()
    val_f1 = evaluate(val_mask)
    if (epoch + 1) % 10 == 0:
        print(f'Epoch {epoch + 1}, Loss: {loss:.4f}, Val F1: {val_f1:.4f}')

# 9. 最终测试
test_f1 = evaluate(test_mask)
print(f'Test F1: {test_f1:.4f}')