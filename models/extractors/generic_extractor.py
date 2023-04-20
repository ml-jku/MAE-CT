import torch.nn as nn

from .base.extractor_base import ExtractorBase
from .base.forward_hook import ForwardHook


class GenericExtractor(ExtractorBase):
    def __init__(self, allow_multiple_outputs=False, **kwargs):
        super().__init__(**kwargs)
        self.allow_multiple_outputs = allow_multiple_outputs

    def to_string(self):
        pooling_str = f"({self.pooling})" if not isinstance(self.pooling, nn.Identity) else ""
        return f"GenericExtractor{pooling_str}"

    def _register_hooks(self, model):
        hook = ForwardHook(
            outputs=self.outputs,
            output_name=self.model_path,
            allow_multiple_outputs=self.allow_multiple_outputs,
        )
        model.register_forward_hook(hook)
        self.hooks.append(hook)
