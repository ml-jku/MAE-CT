from functools import partial

import torch
import torch.nn.functional as F

from distributed.config import get_rank
from distributed.gather import all_gather_grad
from utils.multi_crop_utils import multi_crop_loss
from utils.nnclr_util import find_nn


class NnclrLoss(torch.nn.Module):
    def __init__(self, temperature, transposed=False):
        super().__init__()
        self.temperature = temperature
        self.transposed = transposed

    def forward(self, projected, predicted, ids, queue, queue_ids):
        assert len(projected) == 2 and len(predicted) == 2
        losses = multi_crop_loss(
            projected,
            predicted,
            partial(self._forward, ids=ids, queue=queue, queue_ids=queue_ids),
        )
        return {f"view{i}-view{j}": loss for (i, j), loss in losses.items()}

    def _forward(self, projected, predicted, ids, queue, queue_ids):
        _, nn = find_nn(projected=projected, ids=ids, queue=queue, queue_ids=queue_ids)
        predicted = F.normalize(predicted, dim=-1)
        predicted = all_gather_grad(predicted)

        if self.transposed:
            logits = predicted @ nn.T / self.temperature
        else:
            logits = nn @ predicted.T / self.temperature
        n = nn.size(0)
        rank = get_rank()
        labels = torch.arange(n * rank, n * (rank + 1), device=predicted.device)
        loss = F.cross_entropy(logits, labels)
        return loss


# TODO only needed for old contrastive impl
def nnclr_loss_fn(predicted, nn, temperature, transposed=False):
    # this is redundant (nn is already normalized)
    # normed_nn = F.normalize(nn, dim=-1)
    normed_nn = nn
    normed_predicted = F.normalize(predicted, dim=-1)



    rank = get_rank()
    if transposed:
        normed_nn = all_gather_grad(normed_nn)
        logits = normed_predicted @ normed_nn.T / temperature
        n = normed_predicted.size(0)
    else:
        normed_predicted = all_gather_grad(normed_predicted)
        logits = normed_nn @ normed_predicted.T / temperature
        n = nn.size(0)

    labels = torch.arange(n * rank, n * (rank + 1), device=predicted.device)
    # reduction="none" has large errors with bfloat16
    # loss = F.cross_entropy(logits, labels, reduction="none")
    loss = F.cross_entropy(logits, labels)
    return loss
