from models import model_from_kwargs
from utils.factory import create
from utils.model_utils import get_output_shape_of_model
from ..base.composite_model_base import CompositeModelBase


class BackboneHead(CompositeModelBase):
    def __init__(self, backbone, head, **kwargs):
        super().__init__(**kwargs)
        self.backbone = create(
            backbone,
            model_from_kwargs,
            stage_path_provider=self.stage_path_provider,
            input_shape=self.input_shape,
            update_counter=self.update_counter,
            ctx=self.ctx,
        ).register_extractor_hooks()
        self.latent_shape = get_output_shape_of_model(model=self.backbone, forward_fn=self.backbone.features)
        self.head = create(
            head,
            model_from_kwargs,
            stage_path_provider=self.stage_path_provider,
            input_shape=self.latent_shape,
            output_shape=self.output_shape,
            update_counter=self.update_counter,
            ctx=self.ctx,
        )
        self.ctx.clear()

    @property
    def submodels(self):
        return dict(backbone=self.backbone, head=self.head)

    def forward(self, x, backbone_forward_kwargs=None):
        features = self.backbone.features(x, **(backbone_forward_kwargs or {}))
        result = self.head(features)
        self.ctx.clear()
        return result

    def features(self, x):
        features = self.backbone.features(x)
        self.ctx.clear()
        return features

    def predict(self, x):
        features = self.backbone.features(x)
        result = self.head.predict(features)
        self.ctx.clear()
        return result

    def predict_binary(self, x, head_kwargs=None):
        features = self.features(x)
        result = self.head.predict_binary(features, **(head_kwargs or {}))
        self.ctx.clear()
        return result
