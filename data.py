

import torch
from torch_geometric.data import HeteroData
import pandas as pd
import pickle
import os
from util import  *


def get_data(filename):
    chunk_size = 3000000
    pkl_path = './data.pkl'
    now_path = "./data.csv"
    time1 = '2022/10/01 23:59'
    time2 = '2022/10/15 23:59'
    if os.path.exists(pkl_path):
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        return data
    if os.path.exists(now_path):
        df=pd.read_csv(now_path)
    else:

        chunks = pd.read_csv(filename, chunksize=chunk_size)
        total_df=pd.DataFrame()
        for i,df in enumerate(chunks):
            print(i)
            df1 = df[df['Is Laundering'] == 1].copy()  # label为1的数据
            df2 = df[df['Is Laundering'] == 0].copy()  # label为0的数据
            len_df2=len(df2)
            sample_size = len(df1)
            df3 = df2.sample(n=2*sample_size if  2*sample_size<len_df2 else len_df2, random_state=42)  # random_state保证可重复性
            df4 = pd.concat([df1, df3])
            total_df=pd.concat([total_df, df4])
        df=total_df
        df.sort_index()
        df.to_csv(now_path, index=True)
    data = HeteroData()
    v_types=["bank","account","currency","amount","virtual","timestamp",'format']
    print(max(df["Timestamp"]),min(df['Timestamp']))

    banks=list((set(df['From Bank']) | set(df['To Bank'])))
    max_bank=max(banks)
    nol_banks=[]
    for bank in banks:
        nol_banks.append(bank/max_bank)
    data['bank'].x=torch.tensor(nol_banks).view(-1,1)
    banks_d={}
    for i,bank in enumerate(banks):
        banks_d[bank]=i

    accounts=list((set(df['Account']) | set(df['Account.1'])))
    max_account=len(accounts)
    accounts_hash=[i/max_account for i in range(len(accounts))]
    data['account'].x=torch.tensor(accounts_hash).view(-1,1)
    accounts_d={}
    for i,account in enumerate(accounts):
        accounts_d[account]=i

    currencies=list((set(df['Receiving Currency']) | set(df['Payment Currency'])))
    max_currency=len(currencies)
    currencies_hash=[i/max_currency for i in range(len(currencies))]
    data['currency'].x=torch.tensor(currencies_hash).view(-1,1)
    currencies_d={}
    for i,currency in enumerate(currencies):
        currencies_d[currency]=i

    amounts=list((set(df['Amount Paid']) | set(df['Amount Received'])))
    max_amount=max(amounts)
    nol_amounts=[]
    for amount in amounts:
        nol_amounts.append(amount/max_amount)
    data['amount'].x=torch.tensor(nol_amounts).view(-1,1)
    amounts_d={}
    for i,amount in enumerate(amounts):
        amounts_d[amount]=i

    formats=list(df['Payment Format'])
    max_format=len(formats)
    formats_hash=[i/max_format for i in range(len(formats))]
    data['format'].x=torch.tensor(formats_hash).view(-1,1)
    formats_d={}
    for i,format in enumerate(formats):
        formats_d[format]=i

    timestamps=list(df['Timestamp'])
    max_timestamp=get_timestamp(max(timestamps))-get_timestamp(min(timestamps))
    min_timestamp=get_timestamp(min(timestamps))
    timestamps_hash=[(get_timestamp(i)-min_timestamp)/max_timestamp for i in timestamps]
    data['timestamp'].x=torch.tensor(timestamps_hash).view(-1,1)
    timestamps_d={}
    for i,timestamp in enumerate(timestamps):
        timestamps_d[timestamp]=i

    virtuals=[1]*len(df)
    data['virtual'].x=torch.tensor(virtuals).view(-1,1)
    print(data['virtual'].x.shape)
    data['virtual'].y=torch.tensor(list(df['Is Laundering']))
    print(data['virtual'].y.shape)
    ti_vi_edges=[[],[]]
    fo_vi_edges=[[],[]]
    ac_vi_edges=[[],[]]
    ba_vi_edges=[[],[]]
    cu_vi_edges=[[],[]]
    am_vi_edges=[[],[]]
    vi_am_edges=[[],[]]
    vi_ba_edges=[[],[]]
    vi_ac_edges=[[],[]]
    vi_cu_edges=[[],[]]
    train_mask =list( df['Timestamp'] <= time1)
    print("train_mask", len(train_mask))
    val_mask = list((df['Timestamp'] > time1) & (df['Timestamp'] <= time2))
    print("val_mask", len(val_mask))
    test_mask =list( df['Timestamp']>time2)
    print("test_mask" ,len(test_mask))
    data['virtual'].train_mask=torch.tensor(train_mask)
    data['virtual'].val_mask=torch.tensor(val_mask)
    data['virtual'].test_mask=torch.tensor(test_mask)
    print("df len:",len(df))
    for idx,row in df.iterrows():
        assert type(idx) == type(0), "idx not int"
        break
    print("enter for loop")
    idx=-1
    for _,row in df.iterrows():
        idx+=1
        if idx%100000==0:
            print(idx)
        fb=row['From Bank']
        tb=row['To Bank']
        af=row['Account']
        rc=row['Receiving Currency']
        fc=row['Payment Currency']
        ap=row['Amount Paid']
        ar=row['Amount Received']
        fo=row['Payment Format']
        ti=row['Timestamp']
        at=row['Account.1']
        ti_vi_edges[0].append(timestamps_d[ti])
        ti_vi_edges[1].append(idx)
        fo_vi_edges[0].append(formats_d[fo])
        fo_vi_edges[1].append(idx)
        ac_vi_edges[0].append(accounts_d[af])
        ac_vi_edges[1].append(idx)
        ba_vi_edges[0].append(banks_d[fb])
        ba_vi_edges[1].append(idx)
        cu_vi_edges[0].append(currencies_d[fc])
        cu_vi_edges[1].append(idx)
        am_vi_edges[0].append(amounts_d[ap])
        am_vi_edges[1].append(idx)
        vi_am_edges[0].append(idx)
        vi_am_edges[1].append(amounts_d[ar])
        vi_ba_edges[0].append(idx)
        vi_ba_edges[1].append(banks_d[tb])
        vi_ac_edges[0].append(idx)
        vi_ac_edges[1].append(accounts_d[at])
        vi_cu_edges[0].append(idx)
        vi_cu_edges[1].append(currencies_d[rc])
    print("out for loop")
    data["timestamp","ti_vi","virtual"].edge_index=torch.tensor(ti_vi_edges)
    data["timestamp", "ti_vi", "virtual"].edge_attr=torch.ones((len(ti_vi_edges[0]), 1), dtype=torch.float32)
    data["virtual","ti_vi","timestamp"].edge_index=torch.tensor([ti_vi_edges[1],ti_vi_edges[0]])
    data["virtual","ti_vi", "timestamp"].edge_attr=torch.zeros((len(ti_vi_edges[1]), 1), dtype=torch.float32)

    data['format','fo_vi','virtual'].edge_index=torch.tensor(fo_vi_edges)
    data['format','fo_vi','virtual'].edge_attr = torch.ones((len(fo_vi_edges[0]), 1), dtype=torch.float32)
    data['virtual','fo_vi', 'format'].edge_index=torch.tensor([fo_vi_edges[1],fo_vi_edges[0]])
    data['virtual','fo_vi', 'format'].edge_attr = torch.zeros((len(fo_vi_edges[1]), 1), dtype=torch.float32)

    data['account','ac_vi','virtual'].edge_index=torch.tensor(ac_vi_edges)
    data['account', 'ac_vi', 'virtual'].edge_attr = torch.ones((len(ac_vi_edges[0]), 1), dtype=torch.float32)
    data['virtual', 'ac_vi', 'account'].edge_index = torch.tensor([ac_vi_edges[1],ac_vi_edges[0]])
    data['virtual', 'ac_vi', 'account'].edge_attr = torch.zeros((len(ac_vi_edges[1]), 1), dtype=torch.float32)

    data['bank','ba_vi','virtual'].edge_index=torch.tensor(ba_vi_edges)
    data['bank','ba_vi','virtual'].edge_attr=torch.ones((len(ba_vi_edges[0]), 1), dtype=torch.float32)
    data['virtual','ba_vi','bank'].edge_index=torch.tensor([ba_vi_edges[1],ba_vi_edges[0]])
    data['virtual','ba_vi','bank'].edge_attr=torch.zeros((len(ba_vi_edges[0]), 1), dtype=torch.float32)

    data['currency','cu_vi','virtual'].edge_index=torch.tensor(cu_vi_edges)
    data['currency','cu_vi','virtual'].edge_attr=torch.ones((len(cu_vi_edges[0]), 1), dtype=torch.float32)
    data['virtual','cu_vi','currency'].edge_index=torch.tensor([cu_vi_edges[1],cu_vi_edges[0]])
    data['virtual','cu_vi','currency'].edge_attr=torch.zeros((len(cu_vi_edges[0]), 1), dtype=torch.float32)

    data['amount','am_vi','virtual'].edge_index=torch.tensor(am_vi_edges)
    data['amount','am_vi','virtual'].edge_attr=torch.ones((len(am_vi_edges[0]), 1), dtype=torch.float32)
    data['virtual','am_vi','amount'].edge_index=torch.tensor([am_vi_edges[1],am_vi_edges[0]])
    data['virtual','am_vi','amount'].edge_attr=torch.zeros((len(am_vi_edges[0]), 1), dtype=torch.float32)

    data['virtual','vi_am','amount'].edge_index=torch.tensor(vi_am_edges)
    data['virtual','vi_am','amount'].edge_attr=torch.ones((len(vi_am_edges[0]), 1), dtype=torch.float32)
    data['amount','vi_am','virtual'].edge_index=torch.tensor([vi_am_edges[1],vi_am_edges[0]])
    data['virtual','vi_am','amount'].edge_attr=torch.zeros((len(vi_am_edges[0]), 1), dtype=torch.float32)

    data['virtual','vi_ba','bank'].edge_index=torch.tensor(vi_ba_edges)
    data['virtual','vi_ba','bank'].edge_attr=torch.ones((len(vi_ba_edges[0]), 1), dtype=torch.float32)
    data['bank','vi_ba','virtual'].edge_index=torch.tensor([vi_ba_edges[1],vi_ba_edges[0]])
    data['bank','vi_ba','virtual'].edge_attr=torch.zeros((len(vi_ba_edges[0]),1), dtype=torch.float32)

    data['virtual','vi_ac','account'].edge_index=torch.tensor(vi_ac_edges)
    data['virtual','vi_ac','account'].edge_attr=torch.ones((len(vi_ac_edges[0]), 1), dtype=torch.float32)
    data['account','vi_ac','virtual'].edge_index=torch.tensor([vi_ac_edges[1],vi_ac_edges[0]])
    data['account','vi_ac','virtual'].edge_attr=torch.zeros((len(vi_ac_edges[0]),1), dtype=torch.float32)

    data['virtual','vi_cu','currency'].edge_index=torch.tensor(vi_cu_edges)
    data['virtual','vi_cu','currency'].edge_attr=torch.ones((len(vi_cu_edges[0]),1), dtype=torch.float32)
    data['currency', 'vi_cu', 'virtual'].edge_index = torch.tensor([vi_cu_edges[1],vi_cu_edges[0]])
    data['currency','vi_cu','virtual'].edge_attr=torch.zeros((len(vi_cu_edges[0]),1), dtype=torch.float32)

    data=convert_heterodata_to_float(data)
    with open(pkl_path, 'wb') as f:
        pickle.dump(data, f)  # 序列化对象到二进制文件
    print("out get_data")
    return data


def get_GDSC_data(file_name):
    time1 = 120
    time2 = 160
    # ['Unnamed: 0', 'TX_ID', 'SENDER_ACCOUNT_ID', 'RECEIVER_ACCOUNT_ID','TX_TYPE', 'TX_AMOUNT', 'TIMESTAMP', 'ALERT_ID', 'IS_FRAUD'],
    pkl_path = './GDSC_data.pkl'
    if os.path.exists(pkl_path):
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        return data
    df = pd.read_csv(file_name,nrows=20000)
    data = HeteroData()
    v_types = ['TX_ID', 'SENDER_ACCOUNT_ID', 'RECEIVER_ACCOUNT_ID', 'TX_TYPE', 'TX_AMOUNT', 'TIMESTAMP', 'IS_FRAUD']
    tx_id_str, sender_str, receiver_str, type_str, amount_str, timestamp_str, label_str = (
    v_types[0], v_types[1], v_types[2], v_types[3], v_types[4], v_types[5], v_types[6])
    accounts = list(set(df[sender_str]) | set(df[receiver_str]))
    nol_accounts = []
    max_accounts = len(accounts)
    hash_accounts = {}
    for i, account in enumerate(accounts):
        nol_accounts.append(1.0)
        hash_accounts[account] = i
    data['account'].x = torch.tensor(nol_accounts).view(-1, 1)

    types = list(set(df[type_str]))
    nol_types = []
    max_types = len(types)
    hash_type = {}
    for i, type in enumerate(types):
        nol_types.append(i / max_types)
        hash_type[type] = i
    data['types'].x = torch.tensor(nol_types).view(-1, 1)

    times = list(set(df[timestamp_str]))
    nol_times = []
    max_times = max(df[timestamp_str])
    hash_time = {}
    for i, time in enumerate(times):
        nol_times.append(time / max_times)
        hash_time[time] = i
    data['time'].x = torch.tensor(nol_times).view(-1,1)

    nol_amounts = [i / 10 for i in range(1, 10)]
    data['amount'].x = torch.tensor(nol_amounts).view(-1, 1)

    data['virtual'].x = torch.ones((len(df),), dtype=torch.float).view(-1, 1)
    data['virtual'].y = torch.tensor(list(df[label_str].astype(int)))
    ti_vi_edges1 = [[], []]
    ti_vi_edges2 = [[], []]
    ty_vi_edges1 = [[], []]
    ty_vi_edges2 = [[], []]
    am_vi_edges1 = [[], []]
    am_vi_edges2 = [[], []]
    ac_vi_edges1 = [[], []]
    ac_vi_edges2 = [[], []]
    vi_ac_edges1 = [[], []]
    vi_ac_edges2 = [[], []]
    train_mask = list(df[timestamp_str] <= time1)
    val_mask = list((df[timestamp_str] > time1) & (df[timestamp_str] <= time2))
    test_mask = list(df[timestamp_str] > time2)
    data['virtual'].train_mask = torch.tensor(train_mask)
    data['virtual'].val_mask = torch.tensor(val_mask)
    data['virtual'].test_mask = torch.tensor(test_mask)
    idx = -1
    for _, row in df.iterrows():
        idx += 1
        ti_vi_edges1[0].append(hash_time[row[timestamp_str]])
        ti_vi_edges1[1].append(idx)
        ti_vi_edges2[1].append(hash_time[row[timestamp_str]])
        ti_vi_edges2[0].append(idx)

        ty_vi_edges1[0].append(hash_type[row[type_str]])
        ty_vi_edges1[1].append(idx)
        ty_vi_edges2[1].append(hash_type[row[type_str]])
        ty_vi_edges2[0].append(idx)

        am_vi_edges1[0].append(hash_amount(row[amount_str]))
        am_vi_edges1[1].append(idx)
        am_vi_edges2[1].append(hash_amount(row[amount_str]))
        am_vi_edges2[0].append(idx)

        ac_vi_edges1[0].append(hash_accounts[row[sender_str]])
        ac_vi_edges1[1].append(idx)
        ac_vi_edges2[1].append(hash_accounts[row[sender_str]])
        ac_vi_edges2[0].append(idx)

        vi_ac_edges1[1].append(hash_accounts[row[receiver_str]])
        vi_ac_edges1[0].append(idx)
        vi_ac_edges2[0].append(hash_accounts[row[receiver_str]])
        vi_ac_edges2[1].append(idx)

    data['time', 'ti_vi_1', 'virtual'].edge_index = torch.tensor(ti_vi_edges1)
    data['virtual', 'ti_vi_2', 'time'].edge_index = torch.tensor(ti_vi_edges2)
    data['types', 'ty_vi_1', 'virtual'].edge_index = torch.tensor(ty_vi_edges1)
    data['virtual', 'ty_vi_2', 'types'].edge_index = torch.tensor(ty_vi_edges2)
    data['amount', 'am_vi_1', 'virtual'].edge_index = torch.tensor(am_vi_edges1)
    data['virtual', 'am_vi_2', 'amount'].edge_index = torch.tensor(am_vi_edges2)
    data['account', 'ac_vi_1', 'virtual'].edge_index = torch.tensor(ac_vi_edges1)
    data['virtual', 'ac_vi_2', 'account'].edge_index = torch.tensor(ac_vi_edges2)
    data['virtual', 'vi_ac_1', 'account'].edge_index = torch.tensor(vi_ac_edges1)
    data['account', 'vi_ac_2', 'virtual'].edge_index = torch.tensor(vi_ac_edges2)
    data = convert_heterodata_to_float(data)
    with open(pkl_path, 'wb') as f:
        pickle.dump(data, f)
    return data


if __name__ == '__main__':
    # data = get_data("./data.csv")
    get_GDSC_data('./data/GDSC/dataset.csv')