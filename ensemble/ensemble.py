from ast import literal_eval
from collections import Counter

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import get_label_to_num, list_argmax, num_to_label, softmax


# 조화 평균
def harmonic_ensemble():
    # csv 파일 경로 입력
    a_df = pd.read_csv("./ensemble_data/no.32.csv")
    b_df = pd.read_csv("./ensemble_data/no.43.csv")
    c_df = pd.read_csv("./ensemble_data/no.49.csv")

    a_probs = a_df["probs"]
    b_probs = b_df["probs"]
    c_probs = c_df["probs"]

    a_list = []
    b_list = []
    c_list = []

    for row in range(len(a_probs)):
        a_row = literal_eval(a_probs[row])
        b_row = literal_eval(b_probs[row])
        c_row = literal_eval(c_probs[row])

        a_list.append(a_row)
        b_list.append(b_row)
        c_list.append(c_row)

    ensemble_prob = []
    for x, y, z in zip(a_list, b_list, c_list):
        r = []
        for idx in range(len(x)):
            w_a, w_b, w_c = 1, 1, 1

            numer = np.sum([w_a, w_b, w_c])
            denom = np.sum([w_a * (1 / x[idx]), w_b * (1 / y[idx]), w_c * (1 / z[idx])])
            r.append(numer / denom)
        ensemble_prob.append(r)

    normalized_ensemble_prob = []
    for p in ensemble_prob:
        total = np.sum(p)
        norm = np.divide(p, total)
        normalized_ensemble_prob.append(norm)

    ensemble_pred = []
    for p in normalized_ensemble_prob:
        max_idx = np.argmax(p)
        ensemble_pred.append(max_idx)
    ensemble_pred = num_to_label(ensemble_pred)
    data = {"id": range(len(ensemble_pred)), "pred_label": ensemble_pred, "probs": normalized_ensemble_prob}
    final_df = pd.DataFrame(data)

    final_df["probs"] = final_df["probs"].apply(
        lambda combined_probs: "[" + ", ".join(map(str, combined_probs)) + "]" if combined_probs is not None else None
    )

    # 파일 이름 바꾸기
    final_df.to_csv("./ensemble_output/xxx.csv", index=False)


# F1 가중치 소프트 보팅
def f1_weighted_ensemble():
    # csv 파일 경로 입력
    data_path_list = ["./ensemble_data/no.32.csv", "./ensemble_data/no.43.csv", "./ensemble_data/no.49.csv"]
    # 모델 별 F1 스코어 입력
    score_list = np.array([73.8103, 72.9120, 72.8334])
    score_list = softmax(score_list)
    data_list = [pd.read_csv(data_path) for data_path in data_path_list]
    ensemble = pd.DataFrame({f"probs_{num+1}": data["probs"] for num, data in enumerate(data_list)})

    ensemble_pred = []
    ensemble_prob = []
    ensemble_id = []

    for i in tqdm(range(len(ensemble))):
        probs = [eval(ensemble.iloc[i][f"probs_{j+1}"]) for j in range(len(data_path_list))]

        new_prob = np.sum(np.array(probs) * score_list[:, np.newaxis], axis=0)

        pred, _ = list_argmax(new_prob)

        ensemble_pred.append(pred)
        ensemble_prob.append(new_prob)
        ensemble_id.append(i)

    ensemble_pred = num_to_label(ensemble_pred)
    ensemble = pd.DataFrame({"id": ensemble_id, "pred_label": ensemble_pred, "probs": ensemble_prob})
    ensemble["probs"] = ensemble["probs"].apply(
        lambda combined_probs: "[" + ", ".join(map(str, combined_probs)) + "]" if combined_probs is not None else None
    )
    ensemble.to_csv("./ensemble_output/xxy.csv", index=False)


# 라벨 하드 보팅, probs 최대값
def hard_voting_w_score():
    # csv 파일 경로 입력
    output1 = "./ensemble_data/no.32.csv"
    output2 = "./ensemble_data/no.43.csv"
    output3 = "./ensemble_data/no.49.csv"

    df0 = pd.read_csv(output1)
    df1 = pd.read_csv(output2)
    df2 = pd.read_csv(output3)

    # 모델 별 F1 스코어 입력
    scorelist = [73.8103, 72.8334, 69.6835]
    final_df = df0.copy()
    label_mapping = get_label_to_num()
    labeling = []
    for i in tqdm(range(len(df0))):
        temp0, temp1, temp2 = df0.loc[i], df1.loc[i], df2.loc[i]
        tempscore = [literal_eval(temp0["probs"]), literal_eval(temp1["probs"]), literal_eval(temp2["probs"])]

        labels = [temp0["pred_label"], temp1["pred_label"], temp2["pred_label"]]

        label_count = Counter(labels).most_common()

        if len(label_count) >= 2:
            if label_count[0][1] == label_count[1][1]:
                highest = scorelist.index(max(scorelist))
                winner = labels[highest]

                check = 0
                acur = 0

                for j, label in enumerate(labels):
                    if label == winner and acur < tempscore[j][label_mapping[winner]]:
                        acur = tempscore[j][label_mapping[winner]]
                        check = j

                if check == 1:
                    final_df["probs"][i] = df1["probs"][i]
                elif check == 2:
                    final_df["probs"][i] = df2["probs"][i]

            elif label_count[0][1] > label_count[1][1]:
                winner = label_count[0][0]

                check = 0
                acur = 0

                for j, label in enumerate(labels):
                    if label == winner and acur < tempscore[j][label_mapping[winner]]:
                        acur = tempscore[j][label_mapping[winner]]
                        check = j

                if check == 1:
                    final_df.loc[i, "probs"] = df1.loc[i, "probs"]
                elif check == 2:
                    final_df.loc[i, "probs"] = df2.loc[i, "probs"]

        else:
            winner = label_count[0][0]

            check = 0
            acur = 0
            for j, label in enumerate(labels):
                if label == winner and acur < tempscore[j][label_mapping[winner]]:
                    acur = tempscore[j][label_mapping[winner]]
                    check = j
            if check == 1:
                final_df.loc[i, "probs"] = df1.loc[i, "probs"]
            elif check == 2:
                final_df.loc[i, "probs"] = df2.loc[i, "probs"]

        labeling.append(winner)

    final_df["pred_label"] = labeling

    final_df.to_csv("./ensemble_output/xxz.csv", index=False)


# 원하는 앙상블 방식 내 csv파일 바꾸고 실행
harmonic_ensemble()
# f1_weighted_ensemble()
# hard_voting_w_score()
