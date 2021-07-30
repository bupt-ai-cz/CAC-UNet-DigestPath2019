import torch.nn as nn
import torch
import numpy as np
from ..registry import LOSS

@LOSS.register("BCE")
def bce_loss(logits, labels, weights=None, pos_weights=None):
    labels = make_one_hot(labels, logits.size(1))
    weight = torch.ones([logits.size(0), logits.size(1), logits.size(2), logits.size(3)]).cuda()
    pos_weight = torch.ones([logits.size(1), logits.size(2), logits.size(3)]).cuda()
    if weights and pos_weight:
        for i, (w, p_w) in enumerate(zip(weights, pos_weights)):
            weight[:,i] *= w 
            pos_weight[i] *= p_w 
    criterion = nn.BCEWithLogitsLoss(weight = weight, pos_weight=pos_weight)
    return criterion(logits, labels)

@LOSS.register("BCEWithLogits")
def BCEWithLogits_loss(logits, labels, weights=None, pos_weights=None):
    criterion = nn.BCEWithLogitsLoss()
    return criterion(logits, labels)

@LOSS.register("CrossEntropy")
def ce_loss(logits, labels, ignore_index=255, weights=None, pos_weights=None):
    criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
    return criterion(logits, labels)

def make_one_hot(input, num_classes):
    """Convert class index tensor to one hot encoding tensor.
    Args:
         input: A tensor of shape [N, *]
         num_classes: An int of number of class
    Returns:
        A tensor of shape [N, num_classes, *]
    """
    input = input.unsqueeze(1)
    shape = np.array(input.shape)
    shape[1] = num_classes
    shape = tuple(shape)
    result = torch.zeros(shape).cuda()
    result = result.scatter_(1, input, 1)

    return result
