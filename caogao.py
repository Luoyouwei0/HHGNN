import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv


# 定义GNN模型
class EdgeGNN(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim):
        super(EdgeGNN, self).__init__()

        # 使用GAT层处理有向图
        self.conv1 = GATConv(node_dim, hidden_dim, edge_dim=edge_dim)
        self.conv2 = GATConv(hidden_dim, hidden_dim, edge_dim=edge_dim)

        # 边分类器
        self.edge_classifier = nn.Sequential(
            nn.Linear(2 * hidden_dim+edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, edge_index, edge_attr):

        x1 = F.relu(self.conv1(x, edge_index, edge_attr))
        x2 = F.relu(self.conv2(x1, edge_index, edge_attr))

        row, col = edge_index
        edge_features = torch.cat([x2[row], x2[col], edge_attr], dim=-1)

        # 预测边类型
        return self.edge_classifier(edge_features).squeeze(-1)


# 创建示例数据
def create_example_data():
    str_time="Timestamp"
    str_format='Payment Format'
    str_account1="Account"
    str_account2="Account.1"
    str_bank1="From Bank"
    str_bank2="To Bank"
    str_currency1='Payment Currency'
    str_currency2='Receiving Currency'
    str_amount1='Amount Paid'
    str_amount2='Amount Received'
    str_laundering='Is Laundering'
    import pandas as pd
    df = pd.read_csv('./data.csv')
    accounts = list((set(df[str_account1]) | set(df[str_account2])))
    accounts_d={}
    for i,account in enumerate(accounts):
        accounts_d[account]=i
    x=torch.ones((len(accounts),1),dtype=torch.float32)
    edge_index = [[],[]]
    edge_attr=[]
    edge_label=list(df[str_laundering])
    from util import get_timestamp
    max_time=get_timestamp(max(df[str_time]))-get_timestamp(min(df[str_time]))
    min_time=get_timestamp(min(df[str_time]))
    formats=list(set(df[str_format]))
    format_len=len(formats)
    formats_d={}
    for i,format in enumerate(formats):
        formats_d[format]=i
    banks = list((set(df[str_bank1]) | set(df[str_bank2])))
    bank_len=len(banks)
    banks_d={}
    for i,bank in enumerate(banks):
        banks_d[bank]=i
    max_amount=max(set(df[str_amount1]) | set(df[str_amount2]))
    currencies=list(set(df[str_currency1])|set(df[str_currency2]))
    currency_len=len(currencies)
    currency_d={}
    for i,currency in enumerate(currencies):
        currency_d[currency]=i
    for _, row in df.iterrows():
        edge_index[0].append(accounts_d[row[str_account1]])
        edge_index[1].append(accounts_d[row[str_account2]])
        edge_attr.append([(get_timestamp(row[str_time])-min_time)/max_time,
                          formats_d[row[str_format]]/format_len,
                          banks_d[row[str_bank1]]/bank_len,
                          banks_d[row[str_bank2]]/bank_len,
                          row[str_amount1]/max_amount,
                          row[str_amount2]/max_amount,
                          currency_d[row[str_currency1]]/currency_len,
                          currency_d[row[str_currency2]]/currency_len,
                          ])
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    edge_attr =  torch.tensor(edge_attr,  dtype=torch.float)
    edge_label = torch.tensor(edge_label, dtype=torch.float)

    # 创建PyG数据对象
    print(x.shape,edge_index.shape,edge_attr.shape,edge_label.shape)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=edge_label)
    return data
from sklearn.metrics import f1_score, precision_score, recall_score
def calculate_metrics(y_true, y_pred_prob, threshold=0.5):
    y_pred = (y_pred_prob > threshold).astype(int)
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    return f1, precision, recall
# 训练函数
def train():
    epochs=100
    node_dim=1
    edge_dim=8
    lr = 0.001
    hidden_dim = 16
    data = create_example_data()
    model = EdgeGNN(node_dim=node_dim, edge_dim=edge_dim, hidden_dim=hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    # 训练循环
    for epoch in range(1,1+epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.edge_attr)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                pred_prob = torch.sigmoid(out).numpy()
                f1, precision, recall = calculate_metrics(data.y.numpy(), pred_prob)
            print(f'Epoch {epoch}, Loss: {loss.item():.4f}, '
                  f'F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}')
train()
