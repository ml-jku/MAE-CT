import einops
import torch.nn as nn


class BatchNorm1dViT(nn.BatchNorm1d):
    def forward(self, x):
        # transformer uses (batch_size, seqlen, dim) but BatchNorm1d expects (batch_size, dim, seqlen)
        x = einops.rearrange(x, "b l c -> b c l")
        x = super().forward(x)
        x = einops.rearrange(x, "b c l -> b l c")
        return x
