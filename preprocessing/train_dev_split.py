import pandas as pd
from sklearn.model_selection import train_test_split

# 필수 !!!!
# seed & train data path 확인하기
seed = 3
df = pd.read_csv("data/train_entity_swap.csv")
df_label = df["label"]

train_dist, dev_dist = train_test_split(df, test_size=0.2, train_size=0.8, random_state=seed, stratify=df_label)

col = df.columns
train_dist.to_csv(f"data/train_dist_{seed}.csv", columns=col, index=False)
dev_dist.to_csv(f"data/dev_dist_{seed}.csv", columns=col, index=False)
