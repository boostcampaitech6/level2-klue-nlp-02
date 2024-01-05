import torch
from transformers import AutoTokenizer, TrainingArguments

from dataset.load_data import load_and_process_dataset_for_train
from trainer.trainer import NewTrainer
from utils.compute_metrics import compute_metrics
from utils.get_model import get_model
from utils.set_seed import set_seed


def train():
    # load model and tokenizer
    # MODEL_NAME = "bert-base-uncased"
    MODEL_NAME = "klue/bert-base"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_dataset = load_and_process_dataset_for_train("./data/train_dist_1.csv", tokenizer)
    dev_dataset = load_and_process_dataset_for_train("./data/dev_dist_1.csv", tokenizer)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # 모델 불러오기
    model = get_model(MODEL_NAME, device)

    print(model.config)

    # 사용한 option 외에도 다양한 option들이 있습니다.
    # https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments 참고해주세요.
    training_args = TrainingArguments(
        output_dir="./results",  # output directory
        save_total_limit=5,  # number of total save model.
        save_steps=500,  # model saving step.
        num_train_epochs=20,  # total number of training epochs
        learning_rate=5e-5,  # learning_rate
        per_device_train_batch_size=16,  # batch size per device during training
        per_device_eval_batch_size=16,  # batch size for evaluation
        warmup_steps=500,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
        logging_dir="./logs",  # directory for storing logs
        logging_steps=100,  # log saving step.
        evaluation_strategy="steps",  # evaluation strategy to adopt during training
        # `no`: No evaluation during training.
        # `steps`: Evaluate every `eval_steps`.
        # `epoch`: Evaluate every end of epoch.
        eval_steps=500,  # evaluation step.
        load_best_model_at_end=True,
    )

    # 나중에 loss_function 을 config으로 추가
    trainer = NewTrainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=train_dataset,  # training dataset
        eval_dataset=dev_dataset,  # evaluation dataset
        compute_metrics=compute_metrics,  # define metrics function
    )

    # train model
    trainer.train()
    model.save_pretrained("./best_model")


def main():
    set_seed(42)
    train()


if __name__ == "__main__":
    main()
