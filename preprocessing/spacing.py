import re

from pykospacing import Spacing


def spacing_sent(texts):
    """
    띄어쓰기를 보정합니다.
    """
    spacing = Spacing()
    preprocessed_text = [spacing(text) for text in texts]
    return preprocessed_text


def process_entity(entity_str):
    # 세 번째와 네 번째 작은따옴표 사이의 텍스트 추출
    extracted_text = re.findall(r"'(.*?)'", entity_str)
    if extracted_text:
        # 추출된 텍스트에 모든 공백 제거
        cleaned_text = extracted_text[0].replace(" ", "")
        # 띄어쓰기 보정 적용
        spaced_text = spacing_sent([cleaned_text])[0]
        # 결과를 원래 형식으로 변환
        return entity_str.replace(extracted_text[0], spaced_text)
    else:
        return entity_str


def spacing(dataset):
    for column in ["sentence", "subject_entity", "object_entity"]:
        dataset[column] = dataset[column].apply(process_entity)
    return dataset
