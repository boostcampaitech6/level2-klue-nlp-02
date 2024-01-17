import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from transformers import Trainer


class NewTrainer(Trainer):
    def __init__(self, loss_fn, gamma=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fn = loss_fn
        self.gamma = gamma

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


def loss_fn(loss_fn: str = "FocalLoss", gamma: float = 0, log_flag: bool = True):
    """
    :param loss_fn: implement loss function for training
    :return: loss function module(class)
    """

    if loss_fn == "FocalLoss":
        return FocalLoss(gamma = gamma)


# FocalLoss
class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()
