import torch
import torch_geometric
def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch_geometric.is_xpu_available():
        device = torch.device('xpu')
    else:
        device = torch.device('cpu')
    return device
def get_timestamp(time_str):
    from _datetime import datetime
    begin_time="2022/09/01 00:00"
    bt= datetime.strptime(begin_time,"%Y/%m/%d %H:%M")
    dt = datetime.strptime(time_str, "%Y/%m/%d %H:%M")
    timestamp = (dt -bt).total_seconds()//60
    return timestamp
def hash_amount(amount):
    hash_list=[0,10,100,1000,10000,100000,1000000,10000000,100000000]
    for i in range(len(hash_list)+1):
        if amount < hash_list[i]:
            return i-1
def convert_heterodata_to_float(hetero_data):
    # 遍历所有节点类型
    for node_type in hetero_data.node_types:
        if hasattr(hetero_data[node_type], 'x'):
            hetero_data[node_type].x = hetero_data[node_type].x.float()
    return hetero_data