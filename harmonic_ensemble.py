from ast import literal_eval

import numpy as np
import pandas as pd

np.set_printoptions(linewidth=np.inf)

# 파일 이름 바꾸기
a_df = pd.read_csv("ensemble_data/output_43.csv")
b_df = pd.read_csv("ensemble_data/output_49.csv")
c_df = pd.read_csv("ensemble_data/output_67.csv")
d_df = pd.read_csv("ensemble_data/output_75.csv")
e_df = pd.read_csv("ensemble_data/mina.csv")
# f_df = pd.read_csv('ensemble_data/output_32.csv')

a_probs = a_df["probs"]
b_probs = b_df["probs"]
c_probs = c_df["probs"]
d_probs = d_df["probs"]
e_probs = e_df["probs"]
# f_probs = f_df['probs']

a_list = []
b_list = []
c_list = []
d_list = []
e_list = []
# f_list = []

for row in range(len(a_probs)):
    a_row = literal_eval(a_probs[row])
    b_row = literal_eval(b_probs[row])
    c_row = literal_eval(c_probs[row])
    d_row = literal_eval(d_probs[row])
    e_row = literal_eval(e_probs[row])
    # f_row = literal_eval(f_probs[row])

    a_list.append(a_row)
    b_list.append(b_row)
    c_list.append(c_row)
    d_list.append(d_row)
    e_list.append(e_row)
    # f_list.append(f_row)

ensemble_prob = []
for x, y, z, t, e in zip(a_list, b_list, c_list, d_list, e_list):
    r = []
    for idx in range(len(x)):
        w_a, w_b, w_c, w_d, w_e = 1, 1, 1, 0.8, 1

        numer = np.sum([w_a, w_b, w_c])
        denom = np.sum(
            [w_a * (1 / x[idx]), w_b * (1 / y[idx]), w_c * (1 / z[idx]), w_d * (1 / t[idx]), w_e * (1 / e[idx])]
        )
        r.append(numer / denom)
    ensemble_prob.append(r)

normalized_ensemble_prob = []
for p in ensemble_prob:
    total = np.sum(p)
    norm = np.divide(p, total)
    normalized_ensemble_prob.append(norm)

label_mapping = {
    0: "no_relation",
    1: "org:top_members/employees",
    2: "org:members",
    3: "org:product",
    4: "per:title",
    5: "org:alternate_names",
    6: "per:employee_of",
    7: "org:place_of_headquarters",
    8: "per:product",
    9: "org:number_of_employees/members",
    10: "per:children",
    11: "per:place_of_residence",
    12: "per:alternate_names",
    13: "per:other_family",
    14: "per:colleagues",
    15: "per:origin",
    16: "per:siblings",
    17: "per:spouse",
    18: "org:founded",
    19: "org:political/religious_affiliation",
    20: "org:member_of",
    21: "per:parents",
    22: "org:dissolved",
    23: "per:schools_attended",
    24: "per:date_of_death",
    25: "per:date_of_birth",
    26: "per:place_of_birth",
    27: "per:place_of_death",
    28: "org:founded_by",
    29: "per:religion",
}

new_label_list = []
for p in normalized_ensemble_prob:
    max_idx = np.argmax(p)
    new_label_list.append(label_mapping[max_idx])

data = {"id": range(len(new_label_list)), "pred_label": new_label_list, "probs": normalized_ensemble_prob}
final_df = pd.DataFrame(data)

final_df["probs"] = final_df["probs"].apply(
    lambda combined_probs: "[" + ", ".join(map(str, combined_probs)) + "]" if combined_probs is not None else None
)

# 파일 이름 바꾸기
final_df.to_csv("./ensemble_output/5_data_mina.csv", index=False)
