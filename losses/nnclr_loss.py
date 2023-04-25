from functools import partial

import torch
import torch.nn.functional as F

from distributed.config import get_rank
from distributed.gather import all_gather_grad
from utils.nnclr_util import find_nn


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
