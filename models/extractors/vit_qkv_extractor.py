from utils.formatting_util import summarize_indices_list
from .base.extractor_base import ExtractorBase
from .base.forward_hook import ForwardHook


class VitQKVExtractor(ExtractorBase):
    def __init__(self, block_indices=None, use_next_norm=False, **kwargs):
        super().__init__(**kwargs)
        self.block_indices = block_indices

    def to_string(self):
        return f"BlockExtractor({','.join(summarize_indices_list(self.block_indices))},qkv)"

    def _register_hooks(self, model):
        # If block_indices is None, create a hook on each block
        if self.block_indices is None:
            self.block_indices = list(range(len(model.blocks)))
        else:
            # make negative indices positive (for consistency in name)
            for i in range(len(self.block_indices)):
                if self.block_indices[i] < 0:
                    self.block_indices[i] = len(
                        model.blocks) + self.block_indices[i]
            # remove possible duplicates and sort
            self.block_indices = sorted(list(set(self.block_indices)))

        for block_idx in self.block_indices:
            hook = ForwardHook(
                self.outputs, output_name=f"block{block_idx}_qkv")
            model.blocks[block_idx]._modules["attn"]._modules["qkv"].register_forward_hook(
                hook)
            self.hooks.append(hook)
