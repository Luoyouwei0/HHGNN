# import pandas as pd
# filename="../../DBLP/AMLWorld/dataset/HI-Small_Trans.csv"
txt_file="../../data/AMLWorld/dataset/HI-Small_Patterns.txt"
# df=pd.read_csv(filename)
# part_df=df.head(1000)
# part_df.to_csv('./look_able_AMLWorld.csv')
# print(part_df.columns)
# Index(['Timestamp', 'From Bank', 'Account', 'To Bank', 'Account.1',
#        'Amount Received', 'Receiving Currency', 'Amount Paid',
#        'Payment Currency', 'Payment Format', 'Is Laundering'],
#       dtype='object')
# for column in df.columns:
#     print(len(df[column].unique()),"\t",column)
with open(txt_file, 'r', encoding='utf-8') as infile:
    with open('look_able_AMLWorld.txt', 'w', encoding='utf-8') as outfile:
        for i, line in enumerate(infile):
            if i >= 1000:
                break
            outfile.write(line)
# dict={}
# n=0
# flag=True
# for _, row in df.iterrows():
#     if row["Account"] not in dict:
#         dict[row["Account"]]=row["From Bank"]
#     elif dict[row["Account"]] !=row["From Bank"]:
#         print(dict[row["Account"]],row["From Bank"])
#         n+=1
# print(n)
