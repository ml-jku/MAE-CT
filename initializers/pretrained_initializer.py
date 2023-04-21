from pathlib import Path

import torch

from .base.initializer_base import InitializerBase


class PretrainedInitializer(InitializerBase):
    def __init__(self, weights_file, **kwargs):
        super().__init__(**kwargs)
        assert self.stage_path_provider.model_path is not None
        self.weights_uri = Path(self.stage_path_provider.model_path / weights_file).expanduser()
        assert self.weights_uri.exists() and self.weights_uri.is_file()

    def init_weights(self, model, config_provider=None, summary_provider=None):
        sd = torch.load(self.weights_uri)
        # MAE checkpoints are {"model": state_dict}
        if "model" in sd:
            sd = sd["model"]
        # ibot checkpoints are {"state_dict": state_dict}
        if "state_dict" in sd and len(sd) == 1:
            sd = sd["state_dict"]
            # some checkpoints have the weights for the head included (e.g. ViT-B block mask)
            sd = {
                k: v
                for k, v in sd.items()
                if "head." not in k
            }
        # MSN checkpoints are {"target_encoder": state_dict, "prototypes": prototypes}
        # contrastive heads are also in checkpoint (starting with "module.fc.")
        if "target_encoder" in sd and "prototypes" in sd and len(sd) == 2:
            sd = sd["target_encoder"]
            sd = {
                k[len("module."):] if k.startswith("module.") else k: v
                for k, v in sd.items()
                if "module.fc." not in k
            }
        # MoCoV3 checkpoints are {"epoch": ..., "arch": ..., "state_dict": ..., "best_acc1": ..., "optimizer": ...}
        if len(sd) == 5 and sorted(list(sd.keys())) == ["arch", "best_acc1", "epoch", "optimizer", "state_dict"]:
            sd = sd["state_dict"]
            sd = {
                k[len("module."):] if k.startswith("module.") else k: v
                for k, v in sd.items()
                if "module.head." not in k
            }
        model.load_state_dict(sd)

    @property
    def should_apply_model_specific_initialization(self):
        return False
