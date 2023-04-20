import torch
from torch import nn as nn

from utils.param_checking import to_2tuple
from utils.vit_util import patchify_as_1d


class NonsharedPatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.ModuleList(
            nn.Linear(in_chans * patch_size[0] * patch_size[1], embed_dim)
            for _ in range(self.num_patches)
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0], f"Input image height ({H}) doesn't match model ({self.img_size[0]})."
        assert W == self.img_size[1], f"Input image width ({W}) doesn't match model ({self.img_size[1]})."

        x = patchify_as_1d(x, patch_size=self.patch_size)
        x = torch.stack([self.proj[i](x[:, i]) for i in range(self.num_patches)], dim=1)

        x = self.norm(x)
        return x
