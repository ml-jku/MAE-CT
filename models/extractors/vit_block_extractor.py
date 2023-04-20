from utils.formatting_util import summarize_indices_list
from .base.extractor_base import ExtractorBase
from .base.forward_hook import ForwardHook


class VitBlockExtractor(ExtractorBase):
    def __init__(self, block_indices, use_next_norm=False, **kwargs):
        super().__init__(**kwargs)
        self.block_indices = block_indices
        self.use_next_norm = use_next_norm

    def to_string(self):
        next_norm_str = ",normed" if self.use_next_norm else ""
        return f"BlockExtractor({','.join(summarize_indices_list(self.block_indices))},{self.pooling}{next_norm_str})"

    def _register_hooks(self, model):
        # make negative indices positive (for consistency in name)
        for i in range(len(self.block_indices)):
            if self.block_indices[i] < 0:
                self.block_indices[i] = len(model.blocks) + self.block_indices[i]
        # remove possible duplicates and sort
        self.block_indices = sorted(list(set(self.block_indices)))

        for block_idx in self.block_indices:
            hook = ForwardHook(self.outputs, output_name=f"block{block_idx}")
            if self.use_next_norm:
                if block_idx == len(model.blocks) - 1:
                    # last block uses the "model.norm" as normalization
                    model.norm.register_forward_hook(hook)
                else:
                    # use the norm of the next block
                    model.blocks[block_idx].norm1.register_forward_hook(hook)
            else:
                # use the unnormalized block output
                model.blocks[block_idx].register_forward_hook(hook)
            self.hooks.append(hook)
