import kappaprofiler as kp
import torch
import torch.nn as nn
from torch.nn.functional import normalize

from models import model_from_kwargs
from utils.factory import create_collection
from .mae_vit import MaeVit


class MaeContheadsVit(MaeVit):
    def __init__(self, contrastive_heads=None, decoder=None, **kwargs):
        super().__init__(decoder=decoder, **kwargs)
        if contrastive_heads is not None:
            self.contrastive_heads = create_collection(
                contrastive_heads,
                model_from_kwargs,
                stage_path_provider=self.stage_path_provider,
                update_counter=self.update_counter,
                input_shape=self.encoder.output_shape,
            )
            self.contrastive_heads = nn.ModuleDict(self.contrastive_heads)
        else:
            self.contrastive_heads = {}

    @property
    def submodels(self):
        heads = {f"head.{key}": value for key, value in self.contrastive_heads.items()}
        if self.decoder is not None:
            heads["decoder"] = self.decoder
        return dict(encoder=self.encoder, **heads)

    # noinspection PyMethodOverriding
    def forward(self, x, mask_generator, batch_size):
        outputs = super().forward(x, mask_generator=mask_generator)
        latent_tokens = outputs["latent_tokens"]
        outputs.update(self.forward_heads(latent_tokens=latent_tokens, batch_size=batch_size))
        return outputs

    def forward_heads(self, latent_tokens, batch_size):
        outputs = {}
        view_count = int(len(latent_tokens) / batch_size)
        for head_name, head in self.contrastive_heads.items():
            outputs[head_name] = {}
            # seperate forward pass because of e.g. BatchNorm
            with kp.named_profile_async(head_name):
                for i in range(view_count):
                    start_idx = i * batch_size
                    end_idx = (i + 1) * batch_size
                    head_outputs = head(latent_tokens[start_idx:end_idx])
                    outputs[head_name][f"view{i}"] = head_outputs
        return outputs

    def get_nn_classes(self, x, batch_size, features_kwargs):
        latent_tokens = self.features(x, **features_kwargs)
        head_outputs = self.forward_heads(latent_tokens=latent_tokens, batch_size=batch_size)
        outputs = {}
        for head_name, head in self.contrastive_heads.items():
            projected = head_outputs[head_name]["view0"]["projected"]
            normed_projected = normalize(projected, dim=1)
            similarity_matrix = normed_projected @ head.queue.T
            nn_idx = similarity_matrix.max(dim=1)[1]
            nn_class = head.queue_y[nn_idx]
            outputs[head_name] = nn_class
        return outputs
