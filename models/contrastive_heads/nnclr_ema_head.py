from utils.model_utils import update_ema, copy_params
from .nnclr_noqueue_head import NnclrNoqueueHead


class NnclrEmaHead(NnclrNoqueueHead):
    def __init__(self, target_factor=0.0, **kwargs):
        self.target_projector = None
        super().__init__(**kwargs)
        self.target_factor = target_factor

    def load_state_dict(self, state_dict, strict=True):
        # patch for stage2
        if "target_projector" not in state_dict:
            for key in list(state_dict.keys()):
                if key.startswith("projector."):
                    state_dict[f"target_projector.{key[len('projector.'):]}"] = state_dict[key]
        super().load_state_dict(state_dict=state_dict, strict=strict)

    def _model_specific_initialization(self):
        copy_params(self.projector, self.target_projector)

    def register_components(self, input_dim, output_dim, **kwargs):
        super().register_components(input_dim=input_dim, output_dim=output_dim)
        self.target_projector = self.create_projector(input_dim, self.proj_hidden_dim, output_dim)
        for param in self.target_projector.parameters():
            param.requires_grad = False

    def _forward(self, pooled):
        target_projected = self.target_projector(pooled)
        projected = self.projector(pooled)
        predicted = self.predictor(projected)
        return dict(projected=target_projected, predicted=predicted)

    def after_update_step(self):
        update_ema(self.projector, self.target_projector, self.target_factor)
