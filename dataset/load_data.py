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


def preprocessing_dataset(dataset, method: str = "tem_punt_question"):
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


def load_data(dataset_dir, method: str = "tem_punt_question"):
    """csv 파일을 경로에 맡게 불러 옵니다."""
    pd_dataset = pd.read_csv(dataset_dir)
    dataset = preprocessing_dataset(pd_dataset, method)

    return dataset


def tokenized_dataset(dataset, tokenizer, method: str = "tem_punt_question"):
    """tokenizer에 따라 sentence를 tokenizing 합니다."""
    concat_entity = []

    for e01, e02, t01, t02 in zip(
        dataset["subject_entity"], dataset["object_entity"], dataset["subject_type"], dataset["object_type"]
    ):
        temp = ""
        if method == "em2":  # entity_marker_2
            # [PER]조지 해리슨[/PER]이 쓰고
            temp = f"[{t01}]{e01}[/{t01}] [{t02}]{e02}[/{t02}]"

        elif method == "tem":  # typed_entity_marker
            # <O:PER> 조지 해리슨 </O:PER>이 쓰고
            temp = f"<S:{t01}> {e01} </S:{t01}> <O:{t02}> {e02} </O:{t02}>"

        elif method == "tem_punt":  # typed_entity_marker (punt)
            # # ^ PER ^ 조지 해리슨 #이 쓰고
            temp = f" @ * {t01} * {e01} @ # ^ {t02} ^ {e02} # "

        elif method == "tem_punt_question":  # typed_entity_marker (punt) + 뒤에 질문
            temp = f" 에서 @ * {t01} * {e01} @ 와(과) # ^ {t02} ^ {e02} # 는(은) 무슨 관계?"

        elif method == "question_first_tem_punt":  # 앞에 질문 + typed_entity_marker (punt)
            temp = f" @ * {t01} * {e01} @ # ^ {t02} ^ {e02} # 는 무슨 관계?"

        else:
            temp = e01 + "[SEP]" + e02

        concat_entity.append(temp)

    if method == "tem_punt_question":
        tokenized_sentences = tokenizer(
            list(dataset["sentence"]),
            concat_entity,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
            add_special_tokens=True,
        )

    else:
        tokenized_sentences = tokenizer(
            concat_entity,
            list(dataset["sentence"]),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
            add_special_tokens=True,
        )

    return tokenized_sentences


def load_and_process_dataset_for_train(file_path, tokenizer, method: str = "tem_punt_question"):
    # load dataset
    dataset = load_data(file_path, method)
    labels = label_to_num(dataset["label"].values)

    # tokenizing dataset
    tokenized_data = tokenized_dataset(dataset, tokenizer, method)

    # return dataset for pytorch.
    return RE_Dataset(tokenized_data, labels)


def load_test_dataset(dataset_dir, tokenizer, method: str = "tem_punt_question"):
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
    if subject["end_idx"] < object["start_idx"]:
        # subject가 앞, object가 뒤
        first = sentence[: subject["start_idx"]]
        middle = sentence[subject["end_idx"] + 1 : object["start_idx"]]
        last = sentence[object["end_idx"] + 1 :]

        sub_word = sentence[subject["start_idx"] : subject["end_idx"] + 1]
        obj_word = sentence[object["start_idx"] : object["end_idx"] + 1]

        if method == "em2":  # entity_marker_2
            sub_word = f"[{subject['type']}]" + sub_word + f"[/{subject['type']}]"
            obj_word = f"[{object['type']}]" + obj_word + f"[/{object['type']}]"

            return first + sub_word + middle + obj_word + last

        elif method == "tem":  # typed_entity_marker
            sub_word = f"<S:{subject['type']}> " + sub_word + f" </S:{subject['type']}>"
            obj_word = f"<O:{object['type']}> " + obj_word + f" </O:{object['type']}>"

            return first + sub_word + middle + obj_word + last

        elif (
            method == "tem_punt" or method == "tem_punt_question" or method == "question_first_tem_punt"
        ):  # typed_entity_marker (punt)
            sub_word = f" @ * {subject['type']} * " + sub_word + " @ "
            obj_word = f" # ^ {object['type']} ^ " + obj_word + " # "

            return first + sub_word + middle + obj_word + last

        else:
            pass

    else:
        # object가 앞, subject가 뒤
        first = sentence[: object["start_idx"]]
        middle = sentence[object["end_idx"] + 1 : subject["start_idx"]]
        last = sentence[subject["end_idx"] + 1 :]

        sub_word = sentence[subject["start_idx"] : subject["end_idx"] + 1]
        obj_word = sentence[object["start_idx"] : object["end_idx"] + 1]

        if method == "em2":  # entity_marker_2
            sub_word = f"[{subject['type']}]" + sub_word + f"[/{subject['type']}]"
            obj_word = f"[{object['type']}]" + obj_word + f"[/{object['type']}]"

            return first + obj_word + middle + sub_word + last

        elif method == "tem":  # typed_entity_marker
            sub_word = f"<S:{subject['type']}> " + sub_word + f" </S:{subject['type']}>"
            obj_word = f"<O:{object['type']}> " + obj_word + f" </O:{object['type']}>"

            return first + obj_word + middle + sub_word + last

        elif (
            method == "tem_punt" or method == "tem_punt_question" or method == "question_first_tem_punt"
        ):  # typed_entity_marker (punt)
            sub_word = f" @ * {subject['type']} * " + sub_word + " @ "
            obj_word = f" # ^ {object['type']} ^ " + obj_word + " # "

            return first + obj_word + middle + sub_word + last

        else:
            pass

    return sentence


def token_confirm(dataset, tokenizer, method):
    out_dataset = preprocessing_dataset(dataset, method)
    df = pd.DataFrame()

    df["sentence"] = out_dataset["sentence"]
    df["tokenized"] = out_dataset["sentence"].apply(tokenizer.tokenize)

    df.to_csv("tokenized.csv", index=False)
