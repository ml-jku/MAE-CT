from .base.freezer_base import FreezerBase


class VitBlockFreezer(FreezerBase):
    def __init__(self, block_idxs=None, end_idx=None, freeze_stem_if_block0_is_frozen=True, **kwargs):
        super().__init__(**kwargs)
        assert (block_idxs is not None) ^ (end_idx is not None)
        if block_idxs is not None:
            assert isinstance(block_idxs, list) and all(isinstance(block_idx, int) for block_idx in block_idxs)
        self.block_idxs = block_idxs
        self.end_idx = end_idx
        self.freeze_stem_if_block0_is_frozen = freeze_stem_if_block0_is_frozen

    def __str__(self):
        if (self.block_idxs is None or 0 in self.block_idxs) and self.freeze_stem_if_block0_is_frozen:
            freeze_stem_str = ",freeze_stem_if_block0_is_frozen=True"
        else:
            freeze_stem_str = ""
        return f"{type(self).__name__}(block_idxs={self.block_idxs}{freeze_stem_str})"

    def _change_state(self, model, requires_grad):
        if self.end_idx is not None:
            block_idxs = list(range(0, len(model.blocks)))[:self.end_idx]
        else:
            block_idxs = self.block_idxs
        for block_idx in block_idxs:
            block = model.blocks[block_idx]
            for param in block.parameters():
                param.requires_grad = requires_grad
            # freeze cls_token/patch_embed if block0 is frozen
            if block_idx == 0 and self.freeze_stem_if_block0_is_frozen:
                model.cls_token.requires_grad = requires_grad
                for p in model.patch_embed.parameters():
                    p.requires_grad = requires_grad
                if model.use_learnable_pos_embed:
                    model.pos_embed.requires_grad = requires_grad

    def _before_accumulation_step(self, model):
        if self.is_frozen:
            if self.end_idx is not None:
                block_idxs = list(range(0, len(model.blocks)))[:self.end_idx]
            else:
                block_idxs = self.block_idxs
            for block_idx in block_idxs:
                block = model.blocks[block_idx]
                block.eval()
                if block_idx == 0 and self.freeze_stem_if_block0_is_frozen:
                    model.patch_embed.eval()
