from .base.param_group_modifier import ParamGroupModifier


class LayerwiseLrDecayModifier(ParamGroupModifier):
    def __init__(self, decay):
        self.decay = decay

    def get_properties(self, model, name, param):
        # adapted from BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L33
        # this will split the model into len(blocks) + 2 "layers"
        # stem (patch_embed, cls_token, pos_embed) -> blocks -> last norm
        # this means that the last block will already be decayed
        num_layers = len(model.blocks) + 1
        scales = list(self.decay ** (num_layers - i) for i in range(num_layers))
        if name in ['cls_token', 'pos_embed'] or name.startswith('patch_embed'):
            return dict(lr_scale=scales[0])
        elif name.startswith("block"):
            layer = int(name.split('.')[1]) + 1
            return dict(lr_scale=scales[layer])
        elif name.startswith("norm."):
            # last norm is not scaled (i.e. original learning rate)
            return {}
        else:
            raise NotImplementedError

    def __str__(self):
        return f"{type(self).__name__}(decay={self.decay})"
