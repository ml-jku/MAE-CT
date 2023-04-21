from functools import partial

import einops
import torch
import torch.nn.functional as F

from .base.pooling_base import PoolingBase


class SinglePooling(PoolingBase):
    def __init__(self, kind, **kwargs):
        super().__init__(**kwargs)
        self.kind = kind
        self.pool_fn = self.get_pool_fn(kind=kind, model=self.model)

    def __repr__(self):
        return str(self)

    def __str__(self):
        return self.kind

    def forward(self, x, *_, **__):
        return self.pool_fn(x)

    @staticmethod
    def get_pool_fn(kind, model, **kwargs):
        fn, extracted_kwargs = SinglePooling._get_pool_fn(kind, model)
        if len(kwargs) > 0 or len(extracted_kwargs) > 0:
            fn = partial(fn, **extracted_kwargs, **kwargs)
        return fn

    @staticmethod
    def _get_pool_fn(kind, model):
        if kind is None or kind == "none":
            return torch.nn.Identity(), {}
        if kind in ["cls", "class", "cls_token", "class_token"]:
            return SinglePooling.class_token, {}
        if kind in ["mean", "avg", "average", "mean_patch", "avg_patch", "average_patch"]:
            # TODO figure out a way to pass model to poolings from heads
            n_aux_tokens = 1 if model is None else model.n_aux_tokens
            return SinglePooling.mean_patch, dict(n_aux_tokens=n_aux_tokens)
        if kind in ["mean_all", "avg_all", "average_all"]:
            return SinglePooling.mean_all, {}
        if kind in ["max_patch"]:
            # TODO figure out a way to pass model to poolings from heads
            n_aux_tokens = 1 if model is None else model.n_aux_tokens
            return SinglePooling.max_patch, dict(n_aux_tokens=n_aux_tokens)
        if kind in ["max_all"]:
            return SinglePooling.max_all, {}
        if kind in ["cls_avg", "cls_average", "class_avg", "class_average", "cls_mean", "class_mean"]:
            # TODO figure out a way to pass model to poolings from heads
            n_aux_tokens = 1 if model is None else model.n_aux_tokens
            return SinglePooling.class_average, dict(n_aux_tokens=n_aux_tokens)
        if kind in ["image"]:
            _, img_h, img_w = model.input_shape
            # TODO figure out a way to pass model to poolings from heads
            n_aux_tokens = 1 if model is None else model.n_aux_tokens
            kwargs = dict(
                patch_size=model.patch_size,
                image_height=img_h,
                image_width=img_w,
                n_aux_tokens=n_aux_tokens,
            )
            return SinglePooling.image, kwargs
        if kind in ["concat_class_average"]:
            n_aux_tokens = 1 if model is None else model.n_aux_tokens
            return SinglePooling.concat_class_average, dict(n_aux_tokens=n_aux_tokens)
        if kind in ["all_patches"]:
            n_aux_tokens = 1 if model is None else model.n_aux_tokens
            return SinglePooling.all_patches, dict(n_aux_tokens=n_aux_tokens)
        if kind in ["local_mean_patch_2x2"]:
            return SinglePooling.local_mean_patch_2x2, {}
        if kind in ["local_mean_patch_4x4"]:
            return SinglePooling.local_mean_patch_4x4, {}
        raise NotImplementedError

    @staticmethod
    def all_patches(all_tokens, n_aux_tokens):
        return all_tokens[:, n_aux_tokens:]

    @staticmethod
    def class_token(all_tokens):
        return all_tokens[:, 0]

    @staticmethod
    def mean_patch(all_tokens, n_aux_tokens):
        return all_tokens[:, n_aux_tokens:].mean(dim=1)

    @staticmethod
    def mean_all(all_tokens):
        return all_tokens.mean(dim=1)

    @staticmethod
    def max_patch(all_tokens, n_aux_tokens):
        return all_tokens[:, n_aux_tokens:].max(dim=1)[0]

    @staticmethod
    def max_all(all_tokens):
        return all_tokens.max(dim=1)[0]

    @staticmethod
    def class_average(all_tokens, n_aux_tokens):
        avg = SinglePooling.mean_patch(all_tokens, n_aux_tokens=n_aux_tokens)
        cls = SinglePooling.class_token(all_tokens)
        return (avg + cls) / 2

    @staticmethod
    def concat_class_average(all_tokens, n_aux_tokens):
        avg = SinglePooling.mean_patch(all_tokens, n_aux_tokens=n_aux_tokens)
        cls = SinglePooling.class_token(all_tokens)
        return torch.concat([cls, avg], dim=1)

    @staticmethod
    def image(all_tokens, image_height, image_width, patch_size, n_aux_tokens):
        # transform into image with c=feature_dim h=h_seqlen w=w_seqlen
        patch_tokens = all_tokens[:, n_aux_tokens:]
        patch_height, patch_width = patch_size
        img = einops.rearrange(
            patch_tokens,
            "b (p q) c -> b c p q",
            p=image_height // patch_height,
            q=image_width // patch_width,
        )
        return img

    @staticmethod
    def local_mean_patch_2x2(all_tokens):
        # TODO hacky implementation
        n_aux_tokens = 1
        patch_tokens = all_tokens[:, n_aux_tokens:]
        sqrt = patch_tokens.shape[1] ** 0.5
        assert sqrt.is_integer()
        height_seqlen = width_seqlen = int(sqrt)
        img = einops.rearrange(
            patch_tokens,
            "b (p q) c -> b c p q",
            p=height_seqlen,
            q=width_seqlen,
        )
        pooled = F.avg_pool2d(img, kernel_size=height_seqlen // 2)
        return einops.rearrange(pooled, "b c p q -> b (p q) c")

    @staticmethod
    def local_mean_patch_4x4(all_tokens):
        # TODO hacky implementation
        n_aux_tokens = 1
        patch_tokens = all_tokens[:, n_aux_tokens:]
        sqrt = patch_tokens.shape[1] ** 0.5
        assert sqrt.is_integer()
        height_seqlen = width_seqlen = int(sqrt)
        img = einops.rearrange(
            patch_tokens,
            "b (p q) c -> b c p q",
            p=height_seqlen,
            q=width_seqlen,
        )
        pooled = F.avg_pool2d(img, kernel_size=height_seqlen // 4)
        return einops.rearrange(pooled, "b c p q -> b (p q) c")
