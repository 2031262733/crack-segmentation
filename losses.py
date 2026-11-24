"""Losses and pixel-level metrics for binary crack segmentation."""
import torch
import torch.nn.functional as F


def dice_loss(prob, target, eps=1e-6):
    num = 2 * (prob * target).sum(dim=(1, 2, 3)) + eps
    den = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + eps
    return 1 - (num / den).mean()


def deep_supervision_loss(outputs, target, side_weight=0.5):
    """BCE + Dice on every side output, the fused output gets weight 1.0."""
    loss = 0.0
    for i, logit in enumerate(outputs):
        w = 1.0 if i == len(outputs) - 1 else side_weight
        prob = torch.sigmoid(logit)
        loss = loss + w * (F.binary_cross_entropy_with_logits(logit, target) + dice_loss(prob, target))
    return loss


@torch.no_grad()
def confusion(logit, target, thr=0.5):
    pred = (torch.sigmoid(logit) > thr).float()
    tp = (pred * target).sum().item()
    fp = (pred * (1 - target)).sum().item()
    fn = ((1 - pred) * target).sum().item()
    tn = ((1 - pred) * (1 - target)).sum().item()
    return tp, fp, fn, tn


def scores(tp, fp, fn, eps=1e-6):
    p = tp / (tp + fp + eps)
    r = tp / (tp + fn + eps)
    f1 = 2 * p * r / (p + r + eps)
    iou = tp / (tp + fp + fn + eps)
    return {"precision": p, "recall": r, "f1": f1, "iou": iou}
