from .base.param_group_modifier import ParamGroupModifier


class ExcludePosEmbedFromWDModifier(ParamGroupModifier):
    def get_properties(self, model, name, param):
        if name == "pos_embed":
            return dict(weight_decay=0.0)
        return {}

    def __str__(self):
        return type(self).__name__
