import torch.nn as nn

from utils.vit_util import flatten_2d_to_1d


class Flatten2dTokens(nn.Module):
    @staticmethod
    def forward(x, aux_tokens=None):
        return flatten_2d_to_1d(patch_tokens_as_img=x, aux_tokens=aux_tokens)
