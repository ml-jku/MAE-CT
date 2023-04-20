import torch.nn as nn

from initializers.base.initializer_base import InitializerBase


class TruncNormalInitializer(InitializerBase):
    """
    initialize linear layers with trunc_normal and bias with 0
    with truncation:
    - MAE probing std=1e-2 (https://github.com/facebookresearch/mae/blob/main/main_linprobe.py#L218)
        - they claim to follow MoCoV3 but MoCoV3 uses normal instead of trunc_normal
    - MAE finetuning std=2e-5 (https://github.com/facebookresearch/mae/blob/main/main_finetune.py#L257)
    without truncation (essentially the same as std is too small for values to be outside range [-2, 2]):
    - MoCoV3 probing std=1e-2
    - DINO probing std=1e-2 (https://github.com/facebookresearch/dino/blob/cb711401860da580817918b9167ed73e3eef3dcf/eval_linear.py#L243)
    - BarlowTwins probing std=1e-2
    - VICRegL probing std=1e-2 (https://github.com/facebookresearch/VICRegL/blob/main/evaluate.py#L161)
    - IBOT probing std=1e-2 (https://github.com/bytedance/ibot/blob/main/evaluation/eval_linear.py#L276)
    - IBOT finetuning std=1e-3 (https://github.com/bytedance/ibot/blob/main/evaluation/classification_layer_decay/run_class_finetuning.py#L148)
    """

    def __init__(self, std, **kwargs):
        super().__init__(**kwargs)
        self.std = std

    @property
    def should_apply_model_specific_initialization(self):
        return True

    def init_weights(self, model, **_):
        model.apply(self.apply_fn)
        self.logger.info(f"initialized {type(model).__name__} with weight=trunc_normal(std={self.std}) bias=0")

    def apply_fn(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=self.std)
            nn.init.zeros_(m.bias)
