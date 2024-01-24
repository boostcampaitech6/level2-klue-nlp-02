import pandas as pd

# from clean_special_chars import clean_special_chars
# from drop_japanese import filter_japanese_sentences
# from eda import eda_aug
# from entity_swap import entity_swap
# from filter_duplicates import filter_duplicates
# from spacing import spacing
from train_dev_split import split_stratified
from translate_hanja import translate_hanja

train_og = pd.read_csv("./data/train/train.csv")
test_og = pd.read_csv("./data/test/test_data.csv")

df = train_og.copy()
test_df = test_og.copy()

# 한자 한글로 변환
df = translate_hanja(df)
test_df = translate_hanja(test_df)

split_stratified(df)
split_stratified(test_df)
"""
# 중복 행 제거
df = filter_duplicates(df)

#일본어 포함 행 제거
df = filter_japanese_sentences(df)

# 특수문자 처리
df = clean_special_chars(df)
test_df = clean_special_chars(test_df)

# entity_swap
df = entity_swap(df)

# 띄어쓰기 재정렬
df = spacing(df)

# EDA 데이터 증강
df = eda_aug(df)

#Train_dev_split_ straitified_w_label

"""
