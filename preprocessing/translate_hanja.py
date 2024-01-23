import hanja


def to_hangul(sentence):
    return hanja.translate(sentence, "substitution")


def translate_hanja(dataset):
    dataset["sentence"] = dataset["sentence"].apply(to_hangul)
    dataset["subject_entity"] = dataset["subject_entity"].apply(to_hangul)
    dataset["object_entity"] = dataset["object_entity"].apply(to_hangul)
    return dataset
