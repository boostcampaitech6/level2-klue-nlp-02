import torch.nn as nn
from transformers import Trainer


class NewTrainer(Trainer):
    def __init__(self, loss_fn, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fn = loss_fn

    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        labels = inputs["labels"]
        logits = outputs["logits"]

        if self.loss_fn == "CE":
            loss_function = nn.CrossEntropyLoss()
        loss = loss_function(logits, labels)

        return (loss, outputs) if return_outputs else loss
