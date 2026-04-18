import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter


class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.
    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.
    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """

    def __init__(self, num_classes=0, epsilon=0.1, topk_smoothing=False):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1).cuda()
        self.k = 1 if not topk_smoothing else self.num_classes // 50

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        if self.k > 1:
            topk = torch.argsort(-log_probs)[:, :self.k]
            targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1 - self.epsilon)
            targets += torch.zeros_like(log_probs).scatter_(1, topk, self.epsilon / self.k)
        else:
            targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
            targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).mean(0).sum()
        return loss
