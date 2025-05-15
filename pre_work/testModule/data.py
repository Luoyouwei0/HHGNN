import torch
def make_data():
    node_types=["user","amount"]
    node_feature_dims={"user":1,"amount":1}
    user_feature=[0,1,2,3,4]
    user_feature = torch.tensor([[float(i)] for i in user_feature], dtype=torch.float32)
    amount_feature=[1,1,1,1,1,1,1]
    amount_feature = torch.tensor([[float(i)] for i in amount_feature], dtype=torch.float32)
    x_dict={"user":user_feature,"amount":amount_feature}
    edge_features=[False,False,False,False,False,False,True,False,False]
    # 修改为整数类型
    edge_features = torch.tensor([int(i) for i in edge_features], dtype=torch.long)
    edge_nodes=[[False, False, False, True , False, False, True , True , True ],
                [True , False, False, False, False, True , False, False, True ],
                [True , True , True , False, False, False, True , False, False],
                [False, True , False, True , True , False, False, False, False],
                [False, False, True , False, True , True , False, True , False],
                [False, False, False, False, False, False, False, False, True ],
                [True , False, False, False, False, False, False, False, False],
                [False, True , False, False, False, False, False, False, False],
                [False, False, True , True , False, False, False, False, False],
                [False, False, False, False, True , False, False, False, False],
                [False, False, False, False, False, True , True , False, False],
                [False, False, False, False, False, False, False, True , False]]
    

    nodes = []
    edges = []
    for i in range(len(edge_nodes[0])):
        for j in range(len(edge_nodes)):
            if edge_nodes[j][i]:
                nodes.append(j)
                edges.append(i)
    
    hyperedge_index = torch.tensor([nodes, edges], dtype=torch.long)
    
    return node_types,node_feature_dims,hyperedge_index,edge_features,x_dict



