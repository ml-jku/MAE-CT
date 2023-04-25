from functools import partial

import kappaprofiler as kp
import torch
from kappadata import KDMultiViewWrapper
from torch.distributed import all_gather_object

from datasets.sample_wrappers.multi_view_wrapper import MultiViewWrapper
from datasets.sample_wrappers.multi_crop_wrapper import MultiCropWrapper
from distributed.config import is_distributed, get_world_size
from distributed.distributed_data_parallel import DistributedDataParallel
from distributed.gather import all_gather_nograd
from loggers import logger_from_kwargs
from loggers.base.logger_base import LoggerBase
from utils.amp_utils import get_supported_precision
from utils.factory import create_collection
from utils.model_utils import get_trainable_param_count
from utils.seed import get_random_states
from .functional import (
    calculate_effective_batch_size_per_device,
    calculate_batch_size_and_accumulation_steps,
    calculate_automatic_max_batch_size,
    get_grad_scaler_and_autocast_context
)
from .trainer_interface import TrainerInterface


class TrainerBase(TrainerInterface):
    def __init__(
            self,
            effective_batch_size: int,
            device: str,
            loggers: list = None,
            precision: int = 32,
            max_batch_size: int = None,
            sync_batchnorm: bool = True,
            add_default_loggers: bool = True,
            # find_unused_params should not be set to true if it is not needed (to avoid overhead)
            # but sometimes it is required (e.g. when dynamically freezing/unfreezing parameters)
            # when find_unused_params setting static_graph to true can bring speedup
            find_unused_params: bool = False,
            static_graph: bool = False,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.device: torch.device = torch.device(device)
        self.effective_batch_size = effective_batch_size
        self.loggers = create_collection(
            loggers,
            logger_from_kwargs,
            data_container=self.data_container,
            config_provider=self.config_provider,
            summary_provider=self.summary_provider,
            stage_path_provider=self.stage_path_provider,
        )
        self.max_batch_size = max_batch_size
        self.sync_batchnorm = sync_batchnorm
        self.add_default_loggers = add_default_loggers
        self.find_unused_params = find_unused_params
        self.static_graph = static_graph

        self.precision = get_supported_precision(desired_precision=precision, device=self.device)
        self.grad_scaler, self.autocast_context = get_grad_scaler_and_autocast_context(self.precision, self.device)
        self._update_counter = None

        # check that children only override their implementation methods
        assert type(self).train == TrainerBase.train

    @property
    def update_counter(self):
        return self._update_counter

    def _get_input_shape_from_dataset(self, dataset_key):
        # return_ctx=True is required for e.g. MixWrapper
        sample = self.data_container.get_dataset(dataset_key, mode="x", return_ctx=True)[0][0]
        if isinstance(
                self.data_container.get_dataset(dataset_key),
                (KDMultiViewWrapper, MultiViewWrapper, MultiCropWrapper),
        ):
            return sample[0].shape
        return sample.shape

    @property
    def input_shape(self):
        return self._get_input_shape_from_dataset("train")

    @property
    def output_shape(self):
        raise NotImplementedError

    def get_all_loggers(self, is_train):
        loggers = self.loggers
        if self.add_default_loggers:
            loggers = self.get_default_loggers(is_train=is_train) + loggers
        return loggers

    def state_dict(self):
        if is_distributed():
            random_states_per_device = [None for _ in range(get_world_size())]
            all_gather_object(random_states_per_device, get_random_states())
        else:
            random_states_per_device = [get_random_states()]
        # TODO use this epoch/update/sample state to allow loading from e.g. checkpoint=latest
        return dict(
            random_states=random_states_per_device,
            epoch=self.update_counter.cur_checkpoint.epoch,
            update=self.update_counter.cur_checkpoint.update,
            sample=self.update_counter.cur_checkpoint.sample,
        )

    def initialize_model(self, model):
        train_dataset = self.data_container.get_dataset("train")
        if train_dataset.has_wrapper_type(MultiViewWrapper):
            view_adjusted_effective_batch_size = self.effective_batch_size * train_dataset.n_views
            self.logger.info(
                f"using effective_batch_size {view_adjusted_effective_batch_size} instead of "
                f"{self.effective_batch_size} for initializing model (n_views={train_dataset.n_views})"
            )
        elif train_dataset.has_wrapper_type(MultiCropWrapper):
            n_views = sum(train_dataset.views_per_transform)
            assert n_views >= 2
            if n_views > 2:
                self.logger.warning("using multi crop -> not sure what lr scaling to apply with local crops using *2")
            view_adjusted_effective_batch_size = self.effective_batch_size * 2
            self.logger.info(
                f"using effective_batch_size {view_adjusted_effective_batch_size} instead of "
                f"{self.effective_batch_size} for initializing model "
                f"(views_per_transform={train_dataset.views_per_transform})"
            )
        else:
            view_adjusted_effective_batch_size = self.effective_batch_size
        model.initialize(
            lr_scaler_factor=view_adjusted_effective_batch_size,
            config_provider=self.config_provider,
            summary_provider=self.summary_provider,
        )

    def wrap_ddp(self, model):
        assert model.is_initialized
        if is_distributed():
            if get_trainable_param_count(model) > 0:
                if self.find_unused_params:
                    self.logger.info(f"using find_unused_params=True")
                    if self.static_graph:
                        self.logger.info(f"using static_graph=True")
                else:
                    assert not self.static_graph
                model = DistributedDataParallel(
                    model,
                    find_unused_parameters=self.find_unused_params,
                    static_graph=self.static_graph,
                )
                if model.device != torch.device("cpu") and self.sync_batchnorm:
                    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            else:
                self.logger.info(f"not wrapping into DDP (no trainable parameters)")
        return model

    @kp.profile
    def train(self, model, loggers=None):
        model = self.prepare_model(model)
        loggers = loggers or self.get_all_loggers(is_train=True)
        batch_size, accumulation_steps, train_batches_per_epoch = self.prepare_batch_size(model)
        train_loader = self.get_train_loader(batch_size=batch_size)
        self.call_before_training(
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
        self.call_after_training(model)

    def prepare_model(self, model):
        raise NotImplementedError

    def prepare_batch_size(self, model):
        self.logger.info("------------------")
        self.logger.info("PREPARE TRAINER")
        batch_size, accumulation_steps = self._calculate_batch_size_and_accumulation_steps(model)
        # NOTE: this doesn't append it to the "trainer" object from the yaml (and it also shouldn't)
        self.config_provider["trainer/batch_size"] = batch_size
        self.config_provider["trainer/accumulation_steps"] = accumulation_steps
        # can't do len(train_loader) as this has strange behaviour with DistributedSampler
        train_batches_per_epoch = int(
            len(self.data_container.get_dataset("train"))
            / self.effective_batch_size
            * accumulation_steps
        )
        self.logger.info(
            f"train_batches per epoch: {train_batches_per_epoch} "
            f"(world_size={get_world_size()} batch_size={batch_size})"
        )
        return batch_size, accumulation_steps, train_batches_per_epoch

    def call_before_training(self, model, batch_size, train_dataset, loggers):
        self.logger.info("------------------")
        self.logger.info("BEFORE TRAINING")
        model.eval()
        LoggerBase.call_before_training(
            loggers,
            model=model,
            trainer=self,
            train_dataset=train_dataset,
            update_counter=self.update_counter,
            trainer_batch_size=batch_size,
        )
        self.logger.info("------------------")
        for l in loggers:
            self.logger.info(f"{l.name}({l.to_verbose_interval_string()})")

    def _train(self, model, accumulation_steps, train_loader, train_batches_per_epoch, loggers):
        raise NotImplementedError

    def call_after_training(self, model):
        self.logger.info("------------------")
        self.logger.info("AFTER TRAINING")
        model.eval()
        for l in self.loggers:
            l.after_training(model=model, trainer=self, update_counter=self.update_counter)
        LoggerBase.flush()

    def get_train_loader(self, batch_size, num_workers=None):
        raise NotImplementedError

    def _calculate_batch_size_and_accumulation_steps(self, model):
        self.logger.info(
            f"calculating batch_size and accumulation_steps "
            f"(effective_batch_size={self.effective_batch_size})"
        )
        # calculate effective_batch_size_per_device
        assert self.effective_batch_size % get_world_size() == 0, \
            f"effective_batch_size ({self.effective_batch_size}) needs to be multiple of " \
            f"world_size ({get_world_size()})"
        effective_batch_size_per_device = calculate_effective_batch_size_per_device(
            self.effective_batch_size,
            get_world_size(),
        )
        if model.is_batch_size_dependent:
            self.logger.info("model is batch_size dependent -> disabled possible gradient accumulation")
            return effective_batch_size_per_device, 1
        if model.optim is not None:
            # eval runs don't require an optimizer for the model -> use effective_batch_size
            return effective_batch_size_per_device, 1

        self.logger.info(f"effective_batch_size: {self.effective_batch_size}")
        if is_distributed():
            self.logger.info(f"effective_batch_size_per_device: {effective_batch_size_per_device}")
            self.logger.info(f"world_size: {get_world_size()}")

        if self.max_batch_size is None:
            # calculate max_batch_size
            # TODO
            #  I think optimizer states are initialized on first step, so this is not accurate
            #  (especially for large models)
            self.logger.info("calculating automatic max_batch_size")
            with kp.named_profile_async("automatic_max_batch_size"):
                max_batch_size = calculate_automatic_max_batch_size(
                    train_dataset=self.data_container.get_dataset("train", mode=self.dataset_mode, return_ctx=True),
                    # optim step is only taken on (iter_step + 1) % accumulation_steps == 0
                    train_step_fn=partial(
                        self.calculate_automatic_max_batch_size_step,
                        model,
                        iter_step=0,
                        accumulation_steps=2,
                    ),
                    effective_batch_size_per_device=effective_batch_size_per_device,
                    device=model.device,
                    model=model,
                )
            self.logger.info(f"automatic max_batch_size: {max_batch_size}")
            if is_distributed():
                # check if all devices have the same max_batch_size
                max_batch_sizes = all_gather_nograd(max_batch_size)
                assert all(max_batch_size == mbs for mbs in max_batch_sizes)
        else:
            self.logger.info(f"using provided max_batch_size {self.max_batch_size}")
            max_batch_size = self.max_batch_size

        # calculate batch_size and accumulation_steps
        batch_size, accumulation_steps = calculate_batch_size_and_accumulation_steps(
            effective_batch_size_per_device=effective_batch_size_per_device,
            max_batch_size=max_batch_size,
        )
        self.logger.info(f"batch_size: {batch_size}")
        self.logger.info(f"accumulation_steps: {accumulation_steps}")
        return batch_size, accumulation_steps

    def calculate_automatic_max_batch_size_step(self, *args, **kwargs):
        raise NotImplementedError

    @property
    def dataset_mode(self):
        raise NotImplementedError
