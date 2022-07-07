import torch

def dice_score(pred, gt):
    zero_dsc_sum = 0
    dsc_sum = 0
    zero_count = 0
        
    pred = torch.squeeze(pred)
    gt = torch.squeeze(gt)

    for i in range(len(pred)):
        gt_1 = gt[i]
        pred_1 = pred[i]
        pred_1 = torch.sigmoid(pred_1)
        pred_1 = (pred_1 > 0.5).type(torch.uint8)
        
        tp = ((gt_1 + pred_1) == 2).sum().item()
        tn = ((gt_1 + pred_1) == 0).sum().item()
        fp = ((pred_1 - gt_1) == 1).sum().item()
        fn = ((gt_1 - pred_1) == 1).sum().item()

        if gt_1.sum() == 0:
            zero_count += 1
            if (pred_1>0.5).sum() == 0:
                zero_dsc_sum += 1

        else:
            dsc_sum += 2*tp / (2*tp + fn + fp)

    return (zero_dsc_sum, dsc_sum, zero_count, len(pred)-zero_count)