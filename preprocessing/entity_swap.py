import pandas as pd


def swap_label(df, target_label):
    label_df = df[df["label"] == target_label]
    new_label_df = label_df.copy()
    new_label_df["subject_entity"], new_label_df["object_entity"] = (
        label_df["object_entity"],
        label_df["subject_entity"],
    )

    if target_label == "org:members":
        new_label_df["label"] = "org:member_of"
    elif target_label == "org:member_of":
        new_label_df["label"] = "org:members"
    elif target_label == "per:siblings":
        new_label_df["label"] = "per:siblings"

    return new_label_df


def entity_swap(dataset):
    new_member_df = swap_label(dataset, "org:members")
    new_member_of_df = swap_label(dataset, "org:member_of")
    new_sibling_df = swap_label(dataset, "per:siblings")

    output = pd.concat([dataset, new_member_df, new_member_of_df, new_sibling_df], ignore_index=True).sort_index()
    output["id"] = output.index

    return output
