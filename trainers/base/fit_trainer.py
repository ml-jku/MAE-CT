from loggers.default_loggers.dataset_stats_logger import DatasetStatsLogger
from loggers.default_loggers.param_count_logger import ParamCountLogger
from utils.checkpoint import Checkpoint
from utils.update_counter import UpdateCounter
from .trainer_base import TrainerBase


class FitTrainer(TrainerBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._update_counter = UpdateCounter(
            start_checkpoint=Checkpoint(0, 0, 0),
            end_checkpoint=Checkpoint(1),
            updates_per_epoch=1,
            effective_batch_size=1,
        )
        # fit trainer is not dependent on max_batch_size
        if self.max_batch_size is not None:
            self.effective_batch_size = self.max_batch_size

    @property
    def output_shape(self):
        return None

    def get_default_loggers(self):
        default_kwargs = dict(
            data_container=self.data_container,
            config_provider=self.config_provider,
            stage_path_provider=self.stage_path_provider,
        )
        return [
            DatasetStatsLogger(**default_kwargs),
            ParamCountLogger(**default_kwargs),
        ]

    def prepare_model(self, model):
        model = model.to(self.device)
        self.initialize_model(model)
        model = self.wrap_ddp(model)
        return model

    def get_train_loader(self, batch_size, num_workers=None):
        self.logger.info(f"initializing train dataloader")
        return self.data_container.dataloader_from_key(
            is_train_dataset=False,
            dataset_key="train",
            mode=self.dataset_mode,
            return_ctx=True,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
        )

    def _train(self, model, accumulation_steps, train_loader, train_batches_per_epoch, loggers):
        raise NotImplementedError

    def calculate_automatic_max_batch_size_step(self, *args, **kwargs):
        raise NotImplementedError

    @property
    def dataset_mode(self):
        raise NotImplementedError
