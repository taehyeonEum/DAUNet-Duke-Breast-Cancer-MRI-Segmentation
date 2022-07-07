from scipy.spatial.distance import directed_hausdorff as hd_distance
import torch

def total_score(pred, gt):
    zero_slice_classification_sum = 0
    dsc_sum = 0
    iou_sum = 0
    pixaccu_sum = 0
    precision_sum = 0
    recall_sum = 0
    hd_sum = 0

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
                zero_slice_classification_sum += 1

        else:
            if (pred_1>0.5).sum() != 0: 
                hd_sum += housdorff_distance(pred_1, gt_1)
                precision_sum += tp / (tp + fp)
            dsc_sum += 2*tp / (2*tp + fn + fp)
            iou_sum += tp / (tp + fn + fp)
            pixaccu_sum += (tp + tn) / (tp + tn + fp + fn)
            recall_sum += tp / (tp + fn)

    nonzero_count = len(pred)-zero_count
    return (zero_slice_classification_sum, dsc_sum, iou_sum, pixaccu_sum, precision_sum, recall_sum, hd_sum, zero_count, nonzero_count)

def housdorff_distance(pred, gt):
    u = []
    v = []

    resol = len(pred)
    for i in range(resol):
        for j in range(resol):
            if pred[j, i] == 1: u.append(tuple((j, i)))
            if gt[j, i] == 1: v.append(tuple((j, i)))


    return(max(hd_distance(u, v)[0], hd_distance(v, u)[0]))