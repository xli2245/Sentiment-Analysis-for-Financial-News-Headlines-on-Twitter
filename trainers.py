import torch
from torch import nn
from transformers import Trainer
from monai.losses import DiceCELoss
from my_dice_loss import myDiceLoss


class crossEntropyLossTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


class weightedCrossEntropyLossTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1442.0/1923.0, 1442.0/6178.0]).to("cuda:0"))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


class diceCrossEntropyLossTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = DiceCELoss(to_onehot_y=True, softmax=True)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1).unsqueeze_(1))
        return (loss, outputs) if return_outputs else loss


class customDiceLossTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = myDiceLoss(alpha=0.1)
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss