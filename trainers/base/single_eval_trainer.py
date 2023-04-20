import os

from initializers.resume_initializer import ResumeInitializer
from loggers.base.logger_base import LoggerBase
from trainers import trainer_from_kwargs
from utils.checkpoint import Checkpoint
from utils.factory import create
from .trainer_base import TrainerBase
from .trainer_interface import TrainerInterface

# TODO very similar to EvalTrainer
class SingleEvalTrainer(TrainerInterface):
    def __init__(
            self,
            trainer: TrainerBase,
            device,
            sync_batchnorm: bool,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.trainer: TrainerBase = create(
            trainer,
            trainer_from_kwargs,
            device=device,
            sync_batchnorm=sync_batchnorm,
            data_container=self.data_container,
            config_provider=self.config_provider,
            summary_provider=self.summary_provider,
            stage_path_provider=self.stage_path_provider,
            disable_backward=True,
        )
        self.data_container.run_type = "eval"

    def train(self, model):
        model = self.trainer.prepare_model(model)
        loggers = self.trainer.get_all_loggers(is_train=False)
        batch_size, accumulation_steps, train_batches_per_epoch = self.trainer.prepare_batch_size(model)
        train_loader = self.trainer.get_train_loader(batch_size=batch_size, num_workers=0)
        self.trainer.call_before_training(
            model=model,
            batch_size=batch_size,
            train_dataset=train_loader.dataset,
            loggers=loggers,
        )
        self._train(
            model=model,
            accumulation_steps=accumulation_steps,
            train_loader=train_loader,
            train_batches_per_epoch=train_batches_per_epoch,
            loggers=loggers,
        )
        self.trainer.call_after_training(model)

    def _train(self, model, loggers, train_loader, **_):
        self.logger.info("------------------")
        self.logger.info(f"START TESTING")

        # start logger dataloaders
        LoggerBase.call_start_dataloader_iterators(loggers)

        # call loggers
        model.eval()
        LoggerBase.call_after_update(
            loggers,
            update_counter=self.update_counter,
            effective_batch_size=self.trainer.effective_batch_size,
            trainer=self.trainer,
            model=model,
            train_dataset=train_loader.dataset,
        )
        LoggerBase.call_after_epoch(
            loggers,
            update_counter=self.update_counter,
            effective_batch_size=self.trainer.effective_batch_size,
            trainer=self.trainer,
            model=model,
            train_dataset=train_loader.dataset,
        )
        LoggerBase.flush()

    @property
    def update_counter(self):
        return self.trainer.update_counter

    @property
    def input_shape(self):
        return self.trainer.input_shape

    @property
    def output_shape(self):
        return self.trainer.output_shape

    @property
    def dataset_mode(self):
        return self.trainer.dataset_mode
