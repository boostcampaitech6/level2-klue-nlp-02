import re

import torch
import yaml
from transformers import AutoTokenizer, TrainingArguments
from transformers.integrations import WandbCallback

import wandb
from dataset.load_data import load_and_process_dataset_for_train
from trainer.trainer import NewTrainer
from utils.compute_metrics import compute_metrics
from utils.get_model import get_model
from utils.set_seed import set_seed


def train(configs):
    # 시드 고정
    set_seed(wandb.run.config["seed"])

    # 가독성을 위한 컨픽 지정
    train_path = configs["data"]["train_path"]
    dev_path = configs["data"]["dev_path"]
    output_path = configs["data"]["output_path"]

    MODEL_NAME = wandb.run.config["model_name"]
    saved_name = re.sub("/", "_", MODEL_NAME)
    save_total_limit = configs["model"]["save_total_limit"]
    save_steps = configs["model"]["save_steps"]
    learning_rate = float(wandb.run.config["learning_rate"])
    batch_size = wandb.run.config["batch_size"]
    max_epoch = wandb.run.config["max_epoch"]
    warmup_steps = wandb.run.config["warmup_steps"]
    weight_decay = float(wandb.run.config["weight_decay"])
    evaluation_strategy = wandb.run.config["evaluation_strategy"]
    eval_steps = wandb.run.config["eval_steps"]
    loss_function = wandb.run.config["loss_function"]
    entity_method = wandb.run.config["entity_method"]
    gamma = wandb.run.config["gamma"]
    alpha = wandb.run.config["alpha"]

    logging_dir = configs["log"]["logging_dir"]
    logging_steps = configs["log"]["logging_steps"]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.add_special_tokens = {
        "entity": [
            "[ORG]",
            "[PER]",
            "[POH]",
            "[DAT]",
            "[LOC]",
            "[NOH]",
            "[/ORG]",
            "[/PER]",
            "[/POH]",
            "[/DAT]",
            "[/LOC]",
            "[/NOH]",
            "<S:ORG>",
            "<S:PER>",
            "<S:POH>",
            "<S:DAT>",
            "<S:LOC>",
            "<S:NOH>",
            "</S:ORG>",
            "</S:PER>",
            "</S:POH>",
            "</S:DAT>",
            "</S:LOC>",
            "</S:NOH>",
            "<O:ORG>",
            "<O:PER>",
            "<O:POH>",
            "<O:DAT>",
            "<O:LOC>",
            "<O:NOH>",
            "</O:ORG>",
            "</O:PER>",
            "</O:POH>",
            "</O:DAT>",
            "</O:LOC>",
            "</O:NOH>",
        ]
    }

    train_dataset = load_and_process_dataset_for_train(train_path, tokenizer, entity_method)
    dev_dataset = load_and_process_dataset_for_train(dev_path, tokenizer, entity_method)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # 모델 불러오기
    model = get_model(MODEL_NAME, device)
    model.resize_token_embeddings(len(tokenizer))

    print(model.config)

    # 사용한 option 외에도 다양한 option들이 있습니다.
    # https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments 참고해주세요.
    training_args = TrainingArguments(
        output_dir=output_path,  # output directory
        save_total_limit=save_total_limit,  # number of total save model.
        save_steps=save_steps,  # model saving step.
        num_train_epochs=max_epoch,  # total number of training epochs
        learning_rate=learning_rate,  # learning_rate
        per_device_train_batch_size=batch_size,  # batch size per device during training
        per_device_eval_batch_size=batch_size,  # batch size for evaluation
        warmup_steps=warmup_steps,  # number of warmup steps for learning rate scheduler
        weight_decay=weight_decay,  # strength of weight decay
        logging_dir=logging_dir,  # directory for storing logs
        logging_steps=logging_steps,  # log saving step.
        evaluation_strategy=evaluation_strategy,  # evaluation strategy to adopt during training
        # `no`: No evaluation during training.
        # `steps`: Evaluate every `eval_steps`.
        # `epoch`: Evaluate every end of epoch.
        eval_steps=eval_steps,  # evaluation step.
        load_best_model_at_end=True,
        report_to="wandb",
    )

    # 나중에 loss_function 을 config으로 추가
    trainer = NewTrainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=train_dataset,  # training dataset
        eval_dataset=dev_dataset,  # evaluation dataset
        loss_fn=loss_function,  # loss function customizing
        compute_metrics=compute_metrics,  # define metrics function
        gamma=gamma,
        alpha=alpha,
        callbacks=[WandbCallback()],
    )

    # train model
    trainer.train()
    model.save_pretrained(
        f"{output_path}{saved_name}_{batch_size}_{max_epoch}_{learning_rate}_{loss_function}_{weight_decay}_{entity_method}_{gamma}_{alpha}"
    )


def main(configs):
    wandb.login()
    wandb.init(config=configs)
    run_name = f"{wandb.run.config['model_name']}_{wandb.run.config['batch_size']}_{wandb.run.config['max_epoch']}_{wandb.run.config['learning_rate']}_{wandb.run.config['loss_function']}_{wandb.run.config['weight_decay']}_{wandb.run.config['entity_method']}_{wandb.run.config['gamma']}_{wandb.run.config['alpha']}"
    wandb.run.name = run_name
    train(configs)


if __name__ == "__main__":
    with open("./config/sweep.yaml") as f:
        configs = yaml.safe_load(f)
    main(configs)
