import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from dataset.load_data import load_test_dataset
from utils.get_model import get_model
from utils.labeling import num_to_label
from utils.set_seed import set_seed


def inference(model, tokenized_sent, device):
    """
    test dataset을 DataLoader로 만들어 준 후,
    batch_size로 나눠 model이 예측 합니다.
    """
    dataloader = DataLoader(tokenized_sent, batch_size=16, shuffle=False)
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


def main(args):
    set_seed(42)
    """
    주어진 dataset csv 파일과 같은 형태일 경우 inference 가능한 코드입니다.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # load tokenizer
    Tokenizer_NAME = "klue/bert-base"
    tokenizer = AutoTokenizer.from_pretrained(Tokenizer_NAME)

    # load my model
    MODEL_NAME = args.model_dir  # model dir.
    model = get_model(MODEL_NAME, device)

    # load test datset
    test_dataset_dir = "./data/test_data.csv"
    test_id, Re_test_dataset = load_test_dataset(test_dataset_dir, tokenizer)

    # predict answer
    pred_answer, output_prob = inference(model, Re_test_dataset, device)  # model에서 class 추론
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

    output.to_csv("./prediction/submission.csv", index=False)  # 최종적으로 완성된 예측한 라벨 csv 파일 형태로 저장.
    # 필수!!
    print("---- Finish! ----")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # model dir
    parser.add_argument("--model_dir", type=str, default="./best_model")
    args = parser.parse_args()
    print(args)
    main(args)
