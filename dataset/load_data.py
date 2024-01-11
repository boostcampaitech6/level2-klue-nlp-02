from ast import literal_eval

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


def preprocessing_dataset(dataset, method):
    """처음 불러온 csv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다."""
    subject_entity = []
    object_entity = []
    subject_type = []
    object_type = []
    sents = []

    for sub, obj, sent in zip(dataset["subject_entity"], dataset["object_entity"], dataset["sentence"]):
        sub = literal_eval(sub)
        obj = literal_eval(obj)
        subject_entity.append(sub["word"])
        object_entity.append(obj["word"])
        subject_type.append(sub["type"])
        object_type.append(obj["type"])
        sents.append(make_sentence_mark(method, sent, sub, obj))

    out_dataset = pd.DataFrame(
        {
            "id": dataset["id"],
            "sentence": sents,
            "subject_entity": subject_entity,
            "object_entity": object_entity,
            "subject_type": subject_type,
            "object_type": object_type,
            "label": dataset["label"],
        }
    )
    return out_dataset


def load_data(dataset_dir, method):
    """csv 파일을 경로에 맡게 불러 옵니다."""
    pd_dataset = pd.read_csv(dataset_dir)
    dataset = preprocessing_dataset(pd_dataset, method)

    return dataset


def tokenized_dataset(dataset, tokenizer, method):
    """tokenizer에 따라 sentence를 tokenizing 합니다."""
    concat_entity = []

    for e01, e02, t01, t02 in zip(
        dataset["subject_entity"], dataset["object_entity"], dataset["subject_type"], dataset["object_type"]
    ):
        temp = ""
        if method == "em2":  # entity_marker_2
            # [PER]조지 해리슨[/PER]이 쓰고
            temp = f"[{t01}]{e01}[/{t01}] [SEP] [{t02}]{e02}[/{t02}]"

        elif method == "tem":  # typed_entity_marker
            # <O:PER> 조지 해리슨 </O:PER>이 쓰고
            temp = f"<S:{t01}> {e01} </S:{t01}> [SEP] <O:{t02}> {e02} </O:{t02}>"

        elif method == "tem_punt":  # typed_entity_marker (punt)
            # # ^ PER ^ 조지 해리슨 #이 쓰고
            temp = f" @ * {t01} * {e01} @ [SEP] # ^ {t02} ^ {e02} # "

        elif method == "punt_ori":  # typed entity marker (punt) orginal
            # # ^ PER ^ 조지 해리슨 #이 쓰고 [SEP] 조지 해리슨이 쓰고
            temp = f" @ * {t01} * {e01} @ [SEP] # ^ {t02} ^ {e02} # "

        elif method == "punt_multi":  # multi-sentence prompt
            # 아직 미완
            pass

        else:
            # 조지 해리쓴이 쓰고
            temp = e01 + "[SEP]" + e02

        concat_entity.append(temp)

    tokenized_sentences = tokenizer(
        concat_entity,
        list(dataset["sentence"]),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256,
        add_special_tokens=True,
    )

    return tokenized_sentences


def load_and_process_dataset_for_train(file_path, tokenizer, method):
    # load dataset
    dataset = load_data(file_path, method)
    labels = label_to_num(dataset["label"].values)

    # tokenizing dataset
    tokenized_data = tokenized_dataset(dataset, tokenizer, method)

    # return dataset for pytorch.
    return RE_Dataset(tokenized_data, labels)


def load_test_dataset(dataset_dir, tokenizer, method):
    """
    test dataset을 불러온 후,
    tokenizing 하고, RE_Dateset 객체 생성
    """
    test_dataset = load_data(dataset_dir, method)
    test_label = list(map(int, test_dataset["label"].values))
    # tokenizing dataset
    tokenized_test = tokenized_dataset(test_dataset, tokenizer, method)

    return test_dataset["id"], RE_Dataset(tokenized_test, test_label)


def make_sentence_mark(method, sentence, subject, object):
    print(method)
    if method == "em2":  # entity_marker_2
        sentence = sentence.replace(subject["word"], f"[{subject['type']}]{subject['word']}[/{subject['type']}")
        sentence = sentence.replace(object["word"], f"[{object['type']}]{object['word']}[/{object['type']}]")

    elif method == "tem":  # typed_entity_marker
        sentence = sentence.replace(subject["word"], f"<S:{subject['type']}> {subject['word']} </S:{subject['type']}>")
        sentence = sentence.replace(object["word"], f"<O:{object['type']}> {object['word']} </O:{object['type']}>")

    elif method == "tem_punt":  # typed_entity_marker (punt)
        sentence = sentence.replace(subject["word"], f" @ * {subject['type']} * {subject['word']} @ ")
        sentence = sentence.replace(object["word"], f" # ^ {object['type']} ^ {object['word']} # ")

    elif method == "punt_ori":  # typed entity marker (punt) orginal
        sentence2 = sentence.replace(subject["word"], f" @ * {subject['type']} * {subject['word']} @ ")
        sentence2 = sentence.replace(object["word"], f" # ^ {object['type']} ^ {object['word']} # ")
        sentence = sentence + sentence2

    elif method == "punt_multi":  # multi-sentence prompt
        pass

    else:
        pass

    return sentence
