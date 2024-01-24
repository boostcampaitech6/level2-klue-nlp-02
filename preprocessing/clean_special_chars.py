import re


# 특수문자 처리
def preprocess_clean_punc(text):
    punct_mapping = {
        "'s": "",
        "‧": "",
        "â": "a",
        "ć": "c",
        "š": "s",
        "·": " ",
        "*": "",
        "《": "<",
        "》": ">",
        "〈": "<",
        "〉": ">",
        "㎏": "kg",
        "㈜": "(주)",
        "▲": "",
        "‘": "'",
        "₹": "e",
        "´": "'",
        "°": "",
        "€": "e",
        "™": "tm",
        "√": " sqrt ",
        "×": "x",
        "²": "2",
        "—": "-",
        "–": "-",
        "’": "'",
        "_": "-",
        "`": "'",
        "“": '"',
        "”": '"',
        "“": '"',
        "£": "e",
        "∞": "infinity",
        "θ": "theta",
        "÷": "/",
        "α": "alpha",
        "•": ".",
        "à": "a",
        "−": "-",
        "β": "beta",
        "∅": "",
        "³": "3",
        "π": "pi",
    }
    pattern = re.compile("|".join(re.escape(key) for key in punct_mapping.keys()))

    def replace(match):
        return punct_mapping[match.group(0)]

    # Use the sub function to replace matches in the text
    cleaned_text = pattern.sub(replace, text)

    return cleaned_text.strip()


#
def clean_special_chars(dataset):
    for column in ["sentence", "subject_entity", "object_entity"]:
        if column == "sentence":
            dataset[column] = dataset[column].apply(preprocess_clean_punc)
        else:
            dataset[column] = dataset[column].apply(lambda x: eval(x))
            dataset[column] = dataset[column].apply(
                lambda d: {
                    "word": preprocess_clean_punc(d["word"]),
                    "start_idx": d["start_idx"],
                    "end_idx": d["end_idx"],
                    "type": d["type"],
                }
            )
            dataset[column] = dataset[column].apply(str)
    return dataset
