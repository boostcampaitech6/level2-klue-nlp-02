import re

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from dataset.load_data import load_test_dataset
from utils.get_model import get_model
from utils.labeling import num_to_label
from utils.set_seed import set_seed


def inference(model, tokenized_sent, batch_size, device):
    """
    test dataset을 DataLoader로 만들어 준 후,
    batch_size로 나눠 model이 예측 합니다.
    """
    dataloader = DataLoader(tokenized_sent, batch_size=batch_size, shuffle=False)
    model.eval()
    output_pred = []
    output_prob = []
    for i, data in enumerate(tqdm(dataloader)):
        with torch.no_grad():
            outputs = model(
                input_ids=data["input_ids"].to(device),
                attention_mask=data["attention_mask"].to(device),
                token_type_ids=data["token_type_ids"].to(device),
            )
        logits = outputs[0]
        prob = F.softmax(logits, dim=-1).detach().cpu().numpy()
        logits = logits.detach().cpu().numpy()
        result = np.argmax(logits, axis=-1)

        output_pred.append(result)
        output_prob.append(prob)

    return np.concatenate(output_pred).tolist(), np.concatenate(output_prob, axis=0).tolist()


def predict(configs):
    set_seed(configs["seed"])

    MODEL_NAME = configs["model"]["model_name"]

    entity_method = configs["preprocessing"]["entity_method"]

    output_path = configs["data"]["output_path"]
    predict_path = configs["data"]["predict_path"]
    submission_path = configs["data"]["submission_path"]

    saved_name = re.sub("/", "_", MODEL_NAME)
    batch_size = configs["train"]["batch_size"]
    learning_rate = float(configs["train"]["learning_rate"])
    batch_size = configs["train"]["batch_size"]
    max_epoch = configs["train"]["max_epoch"]
    loss_function = configs["train"]["loss_function"]

    """
    주어진 dataset csv 파일과 같은 형태일 경우 inference 가능한 코드입니다.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # load my model
    model = get_model(
        f"{output_path}{saved_name}_{batch_size}_{max_epoch}_{learning_rate}_{loss_function}_test2", device
    )

    # load test datset
    test_id, Re_test_dataset = load_test_dataset(predict_path, tokenizer, entity_method)

    # predict answer
    pred_answer, output_prob = inference(model, Re_test_dataset, batch_size, device)  # model에서 class 추론
    pred_answer = num_to_label(pred_answer)  # 숫자로 된 class를 원래 문자열 라벨로 변환.

    # make csv file with predicted answer
    # 아래 directory와 columns의 형태는 지켜주시기 바랍니다.
    # 필수!!
    output = pd.DataFrame(
        {
            "id": test_id,
            "pred_label": pred_answer,
            "probs": output_prob,
        }
    )

    output.to_csv(
        f"{submission_path}{saved_name}_{batch_size}_{max_epoch}_{learning_rate}_{loss_function}_test2.csv", index=False
    )  # 최종적으로 완성된 예측한 라벨 csv 파일 형태로 저장.
    # 필수!!
    print("---- Finish! ----")


def main(configs):
    predict(configs)


if __name__ == "__main__":
    with open("config/config.yaml") as f:
        configs = yaml.safe_load(f)

    main(configs)
