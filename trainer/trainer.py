import torch.nn as nn
from transformers import Trainer

from loss import loss_fn


class NewTrainer(Trainer):
    def __init__(self, loss_fn, gamma=2, alpha=0.25, method: str = "tem_punt_question", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fn = loss_fn
        self.gamma = gamma
        self.method = method
        self.alpha = alpha

    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        labels = inputs["labels"]
        logits = outputs["logits"]

        if self.loss_fn == "CE":
            loss_function = nn.CrossEntropyLoss()
        elif self.loss_fn == "FocalLoss":
            loss_function = loss_fn("FocalLoss")

        loss = loss_function(logits, labels)

        return (loss, outputs) if return_outputs else loss
