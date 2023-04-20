import torch
import torch.nn as nn

from initializers.base.initializer_base import InitializerBase


class NormalInitializer(InitializerBase):
    """
    initialize linear layers with normal and bias with 0
    used in:
    - MoCoV3 probing std=1e-2
    - DINO probing std=1e-2
    - BarlowTwins probing std=1e-2
    - VICRegL probing std=1e-2
    - IBOT probing std=1e-2
    """

    def __init__(self, std, seed=None, **kwargs):
        super().__init__(**kwargs)
        self.std = std
        self.seed = seed
        self.cpu_rng = None if seed is None else torch.Generator().manual_seed(seed + 1)
        self.gpu_rng = None if seed is None else torch.Generator(device="cuda:0").manual_seed(seed)

    @property
    def should_apply_model_specific_initialization(self):
        return True

    def init_weights(self, model, **_):
        model.apply(self.apply_fn)
        self.logger.info(f"initialized {type(model).__name__} with weight=trunc_normal(std={self.std}) bias=0")

    def apply_fn(self, m):
        if isinstance(m, nn.Linear):
            if self.seed is None:
                nn.init.normal_(m.weight, std=self.std)
            else:
                with torch.no_grad():
                    if str(m.weight.device) == "cpu":
                        rng = self.cpu_rng
                    elif str(m.weight.device) == "cuda:0":
                        rng = self.gpu_rng
                    else:
                        raise NotImplementedError
                    m.weight.normal_(
                        mean=torch.tensor(0., device=m.weight.device),
                        std=torch.tensor(self.std, device=m.weight.device),
                        generator=rng,
                    )
            nn.init.zeros_(m.bias)
