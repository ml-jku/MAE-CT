import einops
import kappaprofiler as kp

from models import model_from_kwargs
from utils.factory import create
from utils.vit_util import patchify_as_1d, unpatchify_from_1d, patchify_as_2d, unpatchify_from_2d
from ..base.composite_model_base import CompositeModelBase


class MaeVit(CompositeModelBase):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = create(
            encoder,
            model_from_kwargs,
            input_shape=self.input_shape,
            update_counter=self.update_counter,
            stage_path_provider=self.stage_path_provider,
        )
        self.decoder = create(
            decoder,
            model_from_kwargs,
            patch_size=self.encoder.patch_size,
            n_aux_tokens=self.encoder.n_aux_tokens,
            input_shape=self.encoder.output_shape,
            output_shape=self.input_shape,
            update_counter=self.update_counter,
            stage_path_provider=self.stage_path_provider,
        )
        self.latent_shape = self.encoder.embedding_dim
        self.output_shape = self.decoder.output_shape if self.decoder is not None else None

    @property
    def submodels(self):
        return dict(encoder=self.encoder, decoder=self.decoder)

    def encode(self, x, mask_generator, single_mask=False):
        return self.encoder(x, mask_generator=mask_generator, single_mask=single_mask)

    def decode(self, x, ids_restore):
        return self.decoder(x, ids_restore=ids_restore)

    def forward(self, x, mask_generator, single_mask=False):
        with kp.named_profile_async("encode"):
            latent_tokens, mask, ids_restore = self.encode(x, mask_generator=mask_generator, single_mask=single_mask)
        outputs = dict(latent_tokens=latent_tokens)
        if self.decoder is not None:
            with kp.named_profile_async("decode"):
                # for experiment that has encoder without mask and does the masking in the decoder to still have a task
                if self.decoder.mask_generator is not None:
                    x_hat, mask = self.decoder(latent_tokens, ids_restore=ids_restore, mask=mask)
                else:
                    x_hat = self.decode(latent_tokens, ids_restore)
                outputs["x_hat"] = x_hat
        outputs["mask"] = mask
        return outputs

    def features(self, x, mask_generator=None):
        if mask_generator is None:
            return self.encode(x, mask_generator=mask_generator)
        # mask and ids_restore are not needed
        return self.encode(x, mask_generator=mask_generator)[0]

    def mask_x(self, x, mask_generator, single_mask=False):
        patches = patchify_as_2d(x, self.encoder.patch_size)
        _, mask, _ = mask_generator.get_mask(patches, single_mask=single_mask)
        _, _, h, w = patches.shape
        mask = einops.rearrange(mask, "bs (h w) -> bs 1 h w", h=h, w=w)
        return unpatchify_from_2d(patches=patches * (1 - mask), patch_size=self.encoder.patch_size)

    # noinspection PyMethodOverriding
    def reconstruct(self, x, mask_generator, normalize_pixels, single_mask=False):
        all_tokens, mask, ids_restore = self.encoder(x, mask_generator=mask_generator, single_mask=single_mask)
        x_hat = self.decoder(all_tokens, ids_restore)

        # undo norm_pix_loss
        if normalize_pixels:
            target = patchify_as_1d(x, self.encoder.patch_size)
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            x_hat *= (var + 1.0e-6) ** 0.5
            x_hat += mean

        # unpatchify
        imgs = unpatchify_from_1d(
            patches=x_hat * mask.unsqueeze(-1),
            patch_size=self.encoder.patch_size,
            img_shape=self.input_shape,
        )
        return imgs
