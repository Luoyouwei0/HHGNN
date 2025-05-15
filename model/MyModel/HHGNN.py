import torch.nn as nn
from torch_geometric.nn import HypergraphConv
class HHGNN(nn.Module):
    def __init__(self,data):
        self.feature_dim=3
        self.mid_dim=10
        self.hidden_dim=10
        self.output_dim=2
        super().__init__()
        self.projections=nn.ModuleDict({
            t: nn.Linear(data.in_dims[t], self.feature_dim) for t in data.node_types
        })


        self.conv1 = HypergraphConv(self.feature_dim, self.hidden_dim)
        self.conv_relu1 = nn.ReLU()
        self.conv2 = HypergraphConv(self.hidden_dim, self.mid_dim)
        self.conv_relu2 = nn.ReLU()
        self.classify_layer1 =nn.Linear(self.mid_dim, self.classify_dim)
        self.classify_relu1 = nn.ReLU()
        self.classify_layer2 = nn.Linear(self.classify_dim, self.classify_dim)
        self.classify_relu2 = nn.ReLU()
        self.classify_layer3 = nn.Linear(self.classify_dim,self.output_dim)
        self.classify_relu3 = nn.ReLU()


    def forward(self,x):
        hid_dim={t: self.projections[t](self.x_dict[t]) for t in self.x_dict} #x_dict:类型，tensor

        x=self.conv1(x, self.hyperedge_index)
        x=self.conv_relu1(x)
        x=self.conv2(x, self.hyperedge_index)
        x=self.conv_relu2(x)

        

