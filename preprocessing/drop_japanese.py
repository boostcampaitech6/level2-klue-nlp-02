import re


def filter_japanese_sentences(dataset):
    def is_japanese(sentence):
        pattern = re.compile("[가-힣]*[ぁ-んァ-ヶ一-龯]+[가-힣]*")
        return bool(pattern.search(sentence))

    dataset = dataset.drop(dataset[dataset["sentence"].apply(is_japanese)].index)
    return dataset
