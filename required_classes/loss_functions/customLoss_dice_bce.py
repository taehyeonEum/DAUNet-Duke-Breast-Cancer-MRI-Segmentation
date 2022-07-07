import torch
import torch.nn as nn

def customLoss(pred, mask, pos_weight, device, smooth):
    pw = torch.Tensor([pos_weight]).to(device)
    bce_lossFunc = nn.BCEWithLogitsLoss(pos_weight=pw)
    bce_loss = bce_lossFunc(pred, mask)

    pred = torch.sigmoid(pred)
    intersection = (pred * mask).sum(dim=(2, 3))
    union = pred.sum(dim=(2,3)) + mask.sum(dim=(2, 3))
    dice_loss = 1 - (2* ((intersection + smooth) / (union + smooth)) )

    return ( bce_loss.mean(), dice_loss.mean() )
