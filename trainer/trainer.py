import torch.nn as nn
from transformers import Trainer


class NewTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        labels = inputs["labels"]
        logits = outputs["logits"]
        loss_function = nn.CrossEntropyLoss()
        loss = loss_function(logits, labels)

        return (loss, outputs) if return_outputs else loss
