from torch.nn.functional import binary_cross_entropy_with_logits, cross_entropy


def bce_loss(preds, target, **kwargs):
    return binary_cross_entropy_with_logits(preds.squeeze(dim=1), target.float(), **kwargs)