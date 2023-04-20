import kappaprofiler as kp
from torch.cuda.amp import GradScaler
from torch.utils.data import DistributedSampler

from distributed.config import is_managed, get_world_size, get_rank
from initializers import initializer_from_kwargs
from initializers.resume_initializer import ResumeInitializer
from loggers.base.logger_base import LoggerBase
from loggers.default_loggers.dataset_stats_logger import DatasetStatsLogger
from loggers.default_loggers.eta_logger import EtaLogger
from loggers.default_loggers.freezer_logger import FreezerLogger
from loggers.default_loggers.lr_logger import LrLogger
from loggers.default_loggers.online_loss_logger import OnlineLossLogger
from loggers.default_loggers.param_count_logger import ParamCountLogger
from loggers.default_loggers.progress_logger import ProgressLogger
from loggers.default_loggers.train_time_logger import TrainTimeLogger
from utils.checkpoint import Checkpoint
from utils.factory import create
from utils.infinite_batch_sampler import InfiniteBatchSampler
from utils.model_utils import get_paramnames_with_no_gradient
from utils.seed import set_random_states
from utils.update_counter import UpdateCounter
from .early_stopper import EarlyStopper
from .trainer_base import TrainerBase


class SgdTrainer(TrainerBase):
    def __init__(
            self,
            max_epochs=None,
            max_updates=None,
            max_samples=None,
            log_every_n_epochs=None,
            log_every_n_updates=None,
            log_every_n_samples=None,
            early_stopper=None,
            initializer: ResumeInitializer = None,
            disable_backward: bool = False,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.end_checkpoint = Checkpoint(max_epochs, max_updates, max_samples)
        self.log_every_n_epochs = log_every_n_epochs
        self.log_every_n_updates = log_every_n_updates
        self.log_every_n_samples = log_every_n_samples
        self.early_stopper = create(early_stopper, EarlyStopper)
        assert len(self.data_container.train) >= self.effective_batch_size, \
            f"{len(self.data_container.train)}<{self.effective_batch_size}"
        self.updates_per_epoch = int(len(self.data_container.train) / self.effective_batch_size)
        self.disable_backward = disable_backward

        self.initializer = create(
            initializer,
            initializer_from_kwargs,
            stage_path_provider=self.stage_path_provider,
        )
        if self.initializer is None:
            self.start_checkpoint = Checkpoint(epoch=0, update=0, sample=0)
        else:
            self.start_checkpoint = Checkpoint.to_fully_specified_from_fnames(
                ckpt_folder=self.stage_path_provider.get_stage_checkpoint_path(
                    stage_name=self.initializer.stage_name,
                    stage_id=self.initializer.stage_id,
                ),
                ckpt=self.initializer.checkpoint,
            )
        self._update_counter = UpdateCounter(
            start_checkpoint=self.start_checkpoint,
            end_checkpoint=self.end_checkpoint,
            updates_per_epoch=self.updates_per_epoch,
            effective_batch_size=self.effective_batch_size,
        )

    def get_default_loggers(self, is_train):
        default_kwargs = dict(
            data_container=self.data_container,
            config_provider=self.config_provider,
            summary_provider=self.summary_provider,
            stage_path_provider=self.stage_path_provider,
        )
        default_intervals = dict(
            every_n_epochs=self.log_every_n_epochs,
            every_n_updates=self.log_every_n_updates,
            every_n_samples=self.log_every_n_samples,
        )
        default_loggers = [
            DatasetStatsLogger(**default_kwargs),
            ParamCountLogger(**default_kwargs),
        ]
        if is_train:
            default_loggers += [
                ProgressLogger(**default_kwargs, **default_intervals),
                TrainTimeLogger(**default_kwargs, **default_intervals),
            ]
            if not self.disable_backward:
                default_loggers.append(OnlineLossLogger(**default_kwargs, **default_intervals, verbose=True))

        # EtaLogger is pointless in managed runs (managed runs don't have an interactive console)
        if not is_managed() and is_train:
            default_loggers = [EtaLogger(**default_kwargs, **default_intervals)] + default_loggers

        # log every 50 updates (this should not be done based on epoch length because wandb plots will be inconsistent
        # accross different epoch lengths)
        # TODO add parameter for this
        # TODO different batch_sizes result in different every_n_update lengths
        if is_train:
            every_n_updates = 50
            default_loggers.append(LrLogger(**default_kwargs, every_n_updates=every_n_updates))
            default_loggers.append(FreezerLogger(**default_kwargs, every_n_updates=every_n_updates))
            default_loggers.append(OnlineLossLogger(**default_kwargs, every_n_updates=every_n_updates, verbose=False))

        for logger in default_loggers:
            self.logger.info(f"added default {type(logger).__name__}({logger.to_verbose_interval_string()})")
        return default_loggers

    def calculate_automatic_max_batch_size_step(self, *args, **kwargs):
        return self.update(*args, **kwargs)

    def state_dict(self):
        state_dict = super().state_dict()
        if isinstance(self.grad_scaler, GradScaler):
            state_dict["grad_scaler"] = self.grad_scaler.state_dict()
        return state_dict

    def load_state_dict(self, state_dict, load_random_states=True):
        if load_random_states:
            # LEGACY random_state was not stored
            if "random_state" in state_dict:
                random_states = state_dict["random_states"]
                # LEGACY checkpoints before 05.11.2022 don't have random_states per device
                if not isinstance(random_states, list):
                    set_random_states(**state_dict["random_states"])
                else:
                    # LIVE BRANCH
                    if len(random_states) != get_world_size():
                        # if world_size is different than in the checkpoint the whole resuming run will not be deterministic
                        # so don't bother to load any random states
                        self.logger.warning(
                            f"trainer checkpoint has different world_size (ckpt_world_size={len(random_states)} "
                            f"world_size={get_world_size()}) -> can't load random states"
                        )
                    else:
                        cur_rank_random_state = random_states[get_rank()]
                        set_random_states(**cur_rank_random_state)
        else:
            self.logger.info(f"random states are NOT loaded")

        if isinstance(self.grad_scaler, GradScaler):
            if "grad_scaler" not in state_dict:
                self.logger.warning(
                    f"trainer checkpoint has no grad_scaler but current trainer uses {self.precision} precision"
                )
            else:
                self.grad_scaler.load_state_dict(state_dict["grad_scaler"])

    def prepare_model(self, model):
        model = model.to(self.device)
        self.initialize_model(model)
        self.apply_resume_initializer(model)
        # assert model.optim is not None, "can't train a model without optimizer"
        model = self.wrap_ddp(model)
        return model

    def apply_resume_initializer(self, model):
        # initialize model to state where it was resumed from
        if self.initializer is not None:
            self.logger.info("------------------")
            self.logger.info("loading trainer/model state for resuming")
            assert isinstance(self.initializer, ResumeInitializer)
            self.logger.info(
                f"loading state from checkpoint {self.initializer.stage_id}/"
                f"{self.initializer.stage_name}/{self.initializer.checkpoint}"
            )
            self.initializer.init_trainer(self)
            self.initializer.init_weights(model)
            self.initializer.init_optim(model)

    def get_train_loader(self, batch_size, num_workers=None):
        self.logger.info(f"initializing train dataloader")
        return self.data_container.dataloader_from_key(
            is_train_dataset=True,
            dataset_key="train",
            mode=self.dataset_mode,
            return_ctx=True,
            batch_size=batch_size,
            num_workers=num_workers,
            end_checkpoint=self.end_checkpoint,
        )

    def _train(self, model, accumulation_steps, train_loader, train_batches_per_epoch, loggers):
        self.logger.info("------------------")
        self.logger.info(f"START TRAINING")
        is_first_update = True

        assert isinstance(train_loader.batch_sampler, InfiniteBatchSampler)
        self.logger.info("initializing dataloader workers")
        with kp.named_profile("iterator"):
            loader_iter = iter(train_loader)
        self.logger.info("initialized dataloader workers")
        # TODO this is not needed anymore
        iter_time = kp.profiler.last_node.last_time

        while not self.update_counter.is_finished:
            model.optim.schedule_epoch_step(self.update_counter.cur_checkpoint)
            if isinstance(train_loader.batch_sampler.sampler, DistributedSampler):
                train_loader.batch_sampler.sampler.set_epoch(self.update_counter.cur_checkpoint.epoch)

            iter_step = -1
            data_time = 0.
            update_time = 0.
            while True:
                # check end of epoch
                remaining_batches = train_batches_per_epoch - (iter_step + 1)
                if remaining_batches < accumulation_steps:
                    # InfiniteBatchSampler already has the next batches preloaded -> fast forward over these
                    for _ in range(remaining_batches):
                        next(loader_iter)
                    break
                is_last_update_in_epoch = remaining_batches - accumulation_steps < accumulation_steps
                LoggerBase.call_before_every_update(loggers, update_counter=self.update_counter, model=model)
                for _ in range(accumulation_steps):
                    # load next batch
                    with kp.named_profile("data_loading"):
                        batch = next(loader_iter)
                        iter_step += 1
                    if iter_step % accumulation_steps == 0:
                        model.optim.schedule_update_step(self.update_counter.cur_checkpoint)
                        data_time = 0.
                        update_time = 0.
                    data_time += kp.profiler.last_node.last_time
                    # start logger dataloaders after the first batch of the train dataloader is completed
                    LoggerBase.call_start_dataloader_iterators(loggers)
                    LoggerBase.call_before_every_accumulation_step(loggers, model=model)

                    model.train()
                    # update contains implicit cuda synchronization points (.detach().cpu(), .item())
                    with kp.named_profile("update"):
                        losses, update_outputs = self.update(
                            batch=batch,
                            iter_step=iter_step,
                            model=model,
                            accumulation_steps=accumulation_steps,
                            train_dataset=train_loader.dataset,
                        )
                    update_time += kp.profiler.last_node.last_time
                    # log unused parameters
                    if is_first_update:
                        unused_param_names = get_paramnames_with_no_gradient(model)
                        self.logger.info(f"{len(unused_param_names)} unused parameters")
                        for name in unused_param_names:
                            self.logger.info(f"- {name}")
                        is_first_update = False
                    LoggerBase.call_track_after_accumulation_step(
                        loggers,
                        update_counter=self.update_counter,
                        trainer=self,
                        model=model,
                        losses=losses,
                        update_outputs=update_outputs,
                        train_dataset=train_loader.dataset,
                        accumulation_steps=accumulation_steps,
                    )
                    # free references to tensors
                    # noinspection PyUnusedLocal
                    update_outputs = None

                # advance counter
                self.update_counter.add_samples(self.effective_batch_size)
                self.update_counter.next_update()
                if is_last_update_in_epoch:
                    self.update_counter.next_epoch()

                model.eval()
                times = dict(iter_time=iter_time, data_time=data_time, update_time=update_time)
                LoggerBase.call_track_after_update_step(
                    loggers,
                    update_counter=self.update_counter,
                    trainer=self,
                    model=model,
                    times=times,
                    train_dataset=train_loader.dataset,
                )
                iter_time = None  # only track iter_time for first iteration where the iterator is actually created
                logger_info_dict = LoggerBase.call_after_update(
                    loggers,
                    update_counter=self.update_counter,
                    effective_batch_size=self.effective_batch_size,
                    trainer=self,
                    model=model,
                    train_dataset=train_loader.dataset,
                )
                # in case that end_checkpoint is defined via samples/updates
                if self.update_counter.is_finished:
                    break

                # no end of epoch -> flush logs from call_after_update
                if not is_last_update_in_epoch:
                    LoggerBase.flush()

                # check update/sample based early stopping
                if self.early_stopper is not None:
                    should_stop_after_update = self.early_stopper.should_stop_after_update(
                        self.update_counter.cur_checkpoint,
                        logger_info_dict,
                    )
                    if should_stop_after_update:
                        return
                    should_stop_after_sample = self.early_stopper.should_stop_after_sample(
                        self.update_counter.cur_checkpoint,
                        logger_info_dict,
                        effective_batch_size=self.effective_batch_size,
                    )
                    if should_stop_after_sample:
                        return

            logger_info_dict = LoggerBase.call_after_epoch(
                loggers,
                update_counter=self.update_counter,
                effective_batch_size=self.effective_batch_size,
                trainer=self,
                model=model,
                train_dataset=train_loader.dataset,
            )
            LoggerBase.flush()

            # check epoch based early stopping
            if self.early_stopper is not None:
                if self.early_stopper.should_stop_after_epoch(self.update_counter.cur_checkpoint, logger_info_dict):
                    return

    def update(self, model, batch, iter_step, accumulation_steps, train_dataset):
        if iter_step % accumulation_steps == 0:
            with kp.named_profile_async("zero_grad"):
                model.optim.zero_grad()
        model.before_accumulation_step()

        with kp.named_profile_async("forward"):
            with self.autocast_context:
                outputs = self.forward(model, batch, train_dataset)
                losses, outputs = self.get_loss(outputs, model)
            total_loss = losses["total"]
            # make sure get_loss returns loss of shape (batch_size,)
            # assert total_loss.ndim == 1
            total_loss = total_loss.mean() / accumulation_steps

        if not self.disable_backward:
            with kp.named_profile_async("backward"):
                self.grad_scaler.scale(total_loss).backward()

            if (iter_step + 1) % accumulation_steps == 0:
                with kp.named_profile_async("step"):
                    model.optim.step(self.grad_scaler)

        return {k: v.detach().cpu() for k, v in losses.items()}, outputs

    @property
    def dataset_mode(self):
        raise NotImplementedError

    @property
    def output_shape(self):
        raise NotImplementedError

    def forward(self, model, batch, train_dataset):
        raise NotImplementedError

    def get_loss(self, outputs, model):
        raise NotImplementedError
