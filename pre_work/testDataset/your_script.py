import pandas as pd
file_path = '../../data/privateDataset/d1.csv'
df=pd.read_csv(file_path)
part_df=df.head(1000)
# part_df.to_csv('./look_able.csv')
print(part_df.columns)
for column in part_df.columns:
    print(column,len(set(df[column].unique())),f"\t\t\tmax: {max(df[column].unique())}, \t\t\tmin: {min(df[column].unique())}")
print("equal:",(df['merchant_code'] == df['receiving_customer_code']).all())