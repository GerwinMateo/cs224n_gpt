import math
import torch
from torch import nn


class LoRALinear(nn.Module):
    def __init__(self, originalLinear, rank=4, alpha=8):
        super().__init__()
        self.originalLinear = originalLinear
        dIn = originalLinear.in_features
        dOut = originalLinear.out_features
        self.scaling = alpha / rank

        self.loraA = nn.Parameter(torch.empty(dIn, rank))
        self.loraB = nn.Parameter(torch.zeros(rank, dOut))
        nn.init.kaiming_uniform_(self.loraA, a=math.sqrt(5))

        for param in self.originalLinear.parameters():
            param.requires_grad = False

    def forward(self, x):
        baseOutput = self.originalLinear(x)
        loraOutput = (x @ self.loraA @ self.loraB) * self.scaling
        return baseOutput + loraOutput


def applyLora(model, targetModules, rank=4, alpha=8):
    replacements = []
    for name, module in model.named_modules():
        for childName, child in module.named_children():
            if isinstance(child, nn.Linear) and childName in targetModules:
                replacements.append((module, childName, child))

    for parent, childName, original in replacements:
        loraLayer = LoRALinear(original, rank=rank, alpha=alpha)
        setattr(parent, childName, loraLayer)

    for param in model.parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        if 'loraA' in name or 'loraB' in name:
            param.requires_grad = True


def printTrainableParams(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    pct = 100 * trainable / total
    print(f"Trainable params: {trainable:,} / {total:,} ({pct:.2f}%)")
