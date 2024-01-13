import pandas as pd
import torch

from utils.labeling import label_to_num


class RE_Dataset(torch.utils.data.Dataset):
    """Dataset 구성을 위한 class."""

    def __init__(self, pair_dataset, labels):
        self.pair_dataset = pair_dataset
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.pair_dataset.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def preprocessing_dataset(dataset):
    """처음 불러온 csv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다."""
    subject_entity = []
    object_entity = []

    for i, j in zip(dataset["subject_entity"], dataset["object_entity"]):
        i = i[1:-1].split(",")[0].split(":")[1]
        j = j[1:-1].split(",")[0].split(":")[1]

        subject_entity.append(i)
        object_entity.append(j)

    out_dataset = pd.DataFrame(
        {
            "id": dataset["id"],
            "sentence": dataset["sentence"],
            "subject_entity": subject_entity,
            "object_entity": object_entity,
            "label": dataset["label"],
        }
    )
    return out_dataset


def load_data(dataset_dir):
    """csv 파일을 경로에 맡게 불러 옵니다."""
    pd_dataset = pd.read_csv(dataset_dir)
    dataset = preprocessing_dataset(pd_dataset)

    return dataset


# create token_type_ids for BART
def create_token_type_ids(tokenizer, concat, sen):
    token_type_ids = []

    for text, sentence in zip(concat, sen):
        tokens = tokenizer.encode_plus(text, sentence, add_special_tokens=True, return_token_type_ids=True)
        token_type_ids.append(tokens["token_type_ids"])

    return token_type_ids


def tokenized_dataset(dataset, tokenizer):
    """tokenizer에 따라 sentence를 tokenizing 합니다."""
    concat_entity = []

    for e01, e02 in zip(dataset["subject_entity"], dataset["object_entity"]):
        temp = ""
        temp = e01 + "[SEP]" + e02
        concat_entity.append(temp)

    sen = list(dataset["sentence"])

    if not tokenizer.sep_token:
        new_special_token = "[SEP]"
        tokenizer.add_special_tokens({"sep_token": new_special_token})

        create_token_type_ids(tokenizer, concat_entity, sen)

    tokenized_sentences = tokenizer(
        concat_entity,
        sen,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128,
        add_special_tokens=True,
    )
    return tokenized_sentences


def load_and_process_dataset_for_train(file_path, tokenizer):
    # load dataset
    dataset = load_data(file_path)
    labels = label_to_num(dataset["label"].values)

    # tokenizing dataset
    tokenized_data = tokenized_dataset(dataset, tokenizer)

    # return dataset for pytorch.
    return RE_Dataset(tokenized_data, labels)


def load_test_dataset(dataset_dir, tokenizer):
    """
    test dataset을 불러온 후,
    tokenizing 하고, RE_Dateset 객체 생성
    """
    test_dataset = load_data(dataset_dir)
    test_label = list(map(int, test_dataset["label"].values))
    # tokenizing dataset
    tokenized_test = tokenized_dataset(test_dataset, tokenizer)
    return test_dataset["id"], RE_Dataset(tokenized_test, test_label)
