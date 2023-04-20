from .base.checkpoint_initializer import CheckpointInitializer


class PreviousRunInitializer(CheckpointInitializer):
    """
    initializes a model from a checkpoint of a previous run (specified by the stage_id)
    load_optim=False as this is usually used for frozen/pretrained models
    """

    def __init__(self, load_optim=False, **kwargs):
        super().__init__(load_optim=load_optim, **kwargs)
