import einops
import torch

from models.poolings.single_pooling import SinglePooling
from .vit_mae import VitMae


class MaskedEncoder(VitMae):
    # noinspection PyMethodOverriding
    def forward(self, x, mask_generator, single_mask=False):
        if mask_generator is None:
            return super().forward(x)

        # embed patches
        x = self.patch_embed(x)

        # add pos embed
        x = x + self.pos_embed

        # undo patch_embed flattening
        # (patch_embed is set to flatten in order to not need to unflatten in inference/without mask)
        h_len, w_len = self.patch_embed.grid_size
        x = einops.rearrange(x, "b (h w) d -> b d h w", h=h_len, w=w_len)

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = mask_generator.get_mask(x, single_mask=single_mask)

        # append cls token
        if self.cls_token is not None:
            cls_token = einops.repeat(self.cls_token, "1 n_tokens dim -> bs n_tokens dim", bs=len(x))
            x = torch.cat((cls_token, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    # noinspection PyMethodOverriding
    def features(self, x, pool_kind=None, mask_generator=None, single_mask=False):
        if mask_generator is not None:
            encoded, _, _ = self(x, mask_generator=mask_generator, single_mask=single_mask)
            return SinglePooling.get_pool_fn(kind=pool_kind, model=self)(encoded)
        return super().features(x, pool_kind=pool_kind)
