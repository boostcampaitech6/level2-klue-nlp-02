import pandas as pd
from sklearn.model_selection import train_test_split

seed = 1  # 여기 바꿔주기
df = pd.read_csv("data/train.csv")  # train 데이터도 필요에 따라 바꿔주세용
df_label = df["label"]

train_dist, dev_dist = train_test_split(df, test_size=0.2, train_size=0.8, random_state=seed, stratify=df_label)

col = df.columns
train_dist.to_csv(f"data/train_dist_{seed}.csv", columns=col, index=False)
dev_dist.to_csv(f"data/dev_dist_{seed}.csv", columns=col, index=False)
