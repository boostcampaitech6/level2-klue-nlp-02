import pickle

import numpy as np


def list_argmax(array):
    idx, max_value = max(enumerate(array), key=lambda x: x[1])
    return idx, max_value


def num_to_label(labels):
    """
    숫자로 되어 있던 class를 원본 문자열 라벨로 변환 합니다.
    """
    with open("../dataset/dict_num_to_label.pkl", "rb") as f:
        dict_num_to_label = pickle.load(f)
    return [dict_num_to_label[v] for v in labels]


def get_label_to_num():
    """
    숫자로 되어 있던 class를 원본 문자열 라벨로 변환 합니다.
    """
    with open("../dataset/dict_label_to_num.pkl", "rb") as f:
        dict_label_to_num = pickle.load(f)
    return dict_label_to_num


def softmax(score_list):
    exp = np.exp(score_list)
    sum_exp = np.sum(exp)
    output = exp / sum_exp
    return output
