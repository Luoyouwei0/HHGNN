import os.path

import torch.nn.functional as F
from model.MyModel.HGT import *
from data import *
from util import *



path="./data/AMLWorld/dataset/HI-Large_Trans.csv"
gdsc_path="./dataset.csv"
epochs=100
hidden_channels=64
out_channels=2
num_heads=2
num_layers=3
lr=0.002
weight_decay=0.000
model_path="./model.pth"


data=get_GDSC_data(gdsc_path)
model = HGT(hidden_channels=hidden_channels, out_channels=out_channels, num_heads=num_heads, num_layers=num_layers,data=data)
if os.path.exists(model_path):
    model=torch.load("./model.pth", weights_only=False)
device=get_device()
data, model = data.to(device), model.to(device)
# with torch.no_grad():  # Initialize lazy modules.
#     out = model(data.x_dict, data.edge_index_dict)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)



def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x_dict, data.edge_index_dict)
    mask = data['virtual'].train_mask
    num_pos = (data['virtual'].y[mask]== 1).sum().float()
    print("num_pos: ", num_pos)
    num_neg = (data['virtual'].y[mask]== 0).sum().float()
    print("num_neg: ", num_neg)
    weight = torch.tensor([1.0, num_neg / num_pos])
    loss = F.cross_entropy(out[mask], data['virtual'].y[mask],weight=weight,reduction='mean')
    loss.backward()
    optimizer.step()
    return float(loss)
@torch.no_grad()
def test():
    model.eval()
    pred = model(data.x_dict, data.edge_index_dict).argmax(dim=-1)

    accs = []
    f1s=[]
    for split in ['train_mask', 'val_mask', 'test_mask']:
        mask = data['virtual'][split]
        y_true = data['virtual'].y[mask]
        y_pred = pred[mask]
        acc = (y_pred == y_true).sum() / mask.sum()
        accs.append(float(acc))
        from sklearn.metrics import f1_score
        f1 = f1_score(y_true.cpu().numpy(), y_pred.cpu().numpy(),  average='binary')
        f1s.append(float(f1))
    return accs,f1s


def main():
    for epoch in range(1, epochs+1):
        loss = train()
        accs,f1s = test()
        torch.save(model,model_path)
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: acc={accs[0]:.4f}|f1={f1s[0]:.4f}, '
              f'Val: acc={accs[1]:.4f}|f1={f1s[2]:.4f}, Test: acc={accs[2]:.4f}|f1={f1s[2]:.4f}')

if __name__ == '__main__':
    main()