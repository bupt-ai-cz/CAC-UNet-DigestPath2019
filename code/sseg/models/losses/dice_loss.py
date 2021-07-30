import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
from functools import partial
from ..registry import LOSS
from .bce_loss import bce_loss
import pdb

class DiceLoss(nn.Module):
    def __init__(self, smooth=0, eps=1e-7):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.eps = eps

    def forward(self, output, target):
        return 1 - (2 * torch.sum(output * target) + self.smooth) / (
                torch.sum(output) + torch.sum(target) + self.smooth + self.eps)

@LOSS.register("BCE_Dice")
def mixed_dice_bce_loss(output, target, dice_weight=0.2, dice_loss=None,
                        bce_weight=0.9,
                        smooth=1, dice_activation='sigmoid',
                        weights=None, pos_weights=None):

    num_classes = output.size(1)
    # target = target[:, :num_classes, :, :].long()
    if dice_loss is None:
        dice = multiclass_dice_loss(output, target, smooth, dice_activation)
    bce = bce_loss(output, target, weights, pos_weights)

    return dice_weight * dice + bce_weight * bce

@LOSS.register("diceloss")
def multiclass_dice_loss(output, target, smooth=0, activation='sigmoid', ignore_bg=False, weights=None, pos_weights=None):
    """Calculate Dice Loss for multiple class output.

    Args:
        output (torch.Tensor): Model output of shape (N x C x H x W).
        target (torch.Tensor): Target of shape (N x H x W).
        smooth (float, optional): Smoothing factor. Defaults to 0.
        activation (string, optional): Name of the activation function, softmax or sigmoid. Defaults to 'softmax'.

    Returns:
        torch.Tensor: Loss value.

    """
    if activation == 'softmax':
        activation_nn = torch.nn.Softmax2d()
    elif activation == 'sigmoid':
        activation_nn = torch.nn.Sigmoid()
    else:
        raise NotImplementedError('only sigmoid and softmax are implemented')

    loss = 0
    dice = DiceLoss(smooth=smooth)
    output = activation_nn(output)
    num_classes = output.size(1)
    target = make_one_hot(target, num_classes)
    target.data = target.data.float()
    for class_nr in range(1,num_classes) if ignore_bg else range(num_classes):
        loss += dice(output[:, class_nr, :, :], target[:, class_nr, :, :])
    if ignore_bg:
        return loss / (num_classes - 1)
    return loss / num_classes

def where(cond, x_1, x_2):
    cond = cond.long()
    return (cond * x_1) + ((1 - cond) * x_2)


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
