from .base.initializer_base import InitializerBase


class DefaultInitializer(InitializerBase):
    """
    default weight initializer
    used e.g. for multi linear head where a heads with different initializers are attached
    """

    @property
    def should_apply_model_specific_initialization(self):
        return True

    def init_weights(self, model, **_):
        pass
