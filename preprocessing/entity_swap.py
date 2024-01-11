import pandas as pd

df = pd.read_csv("/data/ephemeral/home/level2-klue-nlp-02/data/train.csv")

# org:members & org:member_of
# 각 label 당 dataframe 분리
members_df = df[df["label"] == "org:members"]
member_of_df = df[df["label"] == "org:member_of"]

# entity swap 하고 레이블 바꾸기 (org:members -> org:member_of)
new_member_of_df = members_df.copy()
new_member_of_df["subject_entity"] = members_df["object_entity"].values
new_member_of_df["object_entity"] = members_df["subject_entity"].values
new_member_of_df["label"] = "org:member_of"

# entity swap 하고 레이블 바꾸기 (org:member_of -> org:members)
new_member_df = member_of_df.copy()
new_member_df["subject_entity"] = member_of_df["object_entity"].values
new_member_df["object_entity"] = member_of_df["subject_entity"].values
new_member_df["label"] = "org:members"

# per:siblings
# label의 dataframe 분리
siblings_df = df[df["label"] == "per:siblings"]

# entity swap 하고 레이블 바꾸기 (per:siblings subj <-> per:siblings obj)
new_sibling_df = siblings_df.copy()
new_sibling_df["subject_entity"] = siblings_df["object_entity"].values
new_sibling_df["object_entity"] = siblings_df["subject_entity"].values
new_sibling_df["label"] = "per:siblings"

output = pd.concat([df, new_member_df, new_member_of_df, new_sibling_df], ignore_index=True).sort_index()
output["id"] = output.index

output.to_csv("data/train_entity_swap.csv", index=False)
