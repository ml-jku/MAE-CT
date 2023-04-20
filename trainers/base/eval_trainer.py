import os

from initializers.resume_initializer import ResumeInitializer
from loggers.base.logger_base import LoggerBase
from trainers import trainer_from_kwargs
from utils.checkpoint import Checkpoint
from utils.factory import create
from .trainer_base import TrainerBase
from .trainer_interface import TrainerInterface


class EvalTrainer(TrainerInterface):
    def __init__(
            self,
            trainer: TrainerBase,
            stage_name: str,
            stage_id: str,
            device,
            sync_batchnorm: bool,
            eval_only_last_checkpoint: bool = True,
            every_n_epochs: int = 1,
            every_n_updates: int = 1,
            every_n_samples: int = 1,
            model_info: str = None,
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
        )
        self.stage_name = stage_name
        self.stage_id = stage_id
        self.data_container.run_type = "eval"
        self.eval_only_last_checkpoint = eval_only_last_checkpoint
        self.every_n_epochs = every_n_epochs
        self.every_n_updates = every_n_updates
        self.every_n_samples = every_n_samples
        self.model_info = model_info
        if self.eval_only_last_checkpoint:
            assert self.every_n_epochs == 1 and self.every_n_updates == 1 and self.every_n_samples == 1

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

        # print checkpoint files
        ckpt_path = self.stage_path_provider.get_stage_checkpoint_path(
            stage_name=self.stage_name,
            stage_id=self.stage_id,
        )
        files_in_ckpt_folder = os.listdir(ckpt_path)
        self.logger.info(f"checkpoint folder '{ckpt_path}' contains {len(files_in_ckpt_folder)} files")
        for file_in_ckpt_folder in files_in_ckpt_folder:
            self.logger.info(f"- {file_in_ckpt_folder}")

        if self.eval_only_last_checkpoint:
            ckpts = ["last"]
        else:
            # get checkpoints from filenames
            ckpts = [
                Checkpoint.from_filename(fname)
                for fname in files_in_ckpt_folder
                if Checkpoint.contains_checkpoint_string(fname)
            ]
            # filter out duplicates
            ckpts = sorted(list(set(ckpts)))
            if len(ckpts) == 0:
                # try last checkpoint
                last_cp_files = [fname for fname in files_in_ckpt_folder if " cp=last " in fname]
                if len(last_cp_files) > 0:
                    ckpts.append("last")
            # filter out "every_n_..."
            ckpts = [ckpt for ckpt in ckpts if ckpt.epoch % self.every_n_epochs == 0]
            ckpts = [ckpt for ckpt in ckpts if ckpt.update % self.every_n_updates == 0]
            ckpts = [ckpt for ckpt in ckpts if ckpt.sample % self.every_n_samples == 0]
        self.logger.info(f"evaluating checkpoints:")
        for ckpt in ckpts:
            self.logger.info(f"- {ckpt}")


        for ckpt in ckpts:
            # start logger dataloaders
            LoggerBase.call_start_dataloader_iterators(loggers)

            # adjust update_counter to checkpoint
            if isinstance(ckpt, Checkpoint):
                self.update_counter.cur_checkpoint.epoch = ckpt.epoch
                self.update_counter.cur_checkpoint.update = ckpt.update
                self.update_counter.cur_checkpoint.sample = ckpt.sample
            else:
                # currently only supported to evaluate the last checkpoint if checkpoint is not epoch/update/sample
                assert ckpt == "last"

            # load checkpoint
            self.logger.info(
                f"loading model from checkpoint '{ckpt}' "
                f"(stage_name={self.stage_name}, stage_id={self.stage_id})"
            )
            initializer = ResumeInitializer(
                stage_name=self.stage_name,
                stage_id=self.stage_id,
                checkpoint=ckpt,
                load_optim=False,
                load_random_states=False,
                stage_path_provider=self.stage_path_provider,
                model_info=self.model_info,
            )
            initializer.init_weights(model)

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
