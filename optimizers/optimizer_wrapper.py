import logging

import torch

from schedules import schedule_from_schedule_configs
from utils.bidict import Bidict
from utils.factory import create, create_collection
from utils.formatting_util import float_to_scientific_notation
from .lr_scalers import lr_scaler_from_kwargs
from .param_group_modifiers import param_group_modifier_from_kwargs


class OptimizerWrapper:
    """
    wrapper for torch optimizers that also handles
    - learning rate scaling (with batchsize)
    - creating parameter groups (e.g. excluding bias/norm from weight decay, layerwise lr scaling)
    - stateless learning rate scheduling
    - gradient clipping
    """

    def __init__(
            self,
            model,
            torch_optim_ctor,
            schedule=None,
            weight_decay_schedule=None,
            lr_scaler=None,
            lr_scaler_factor=None,
            clip_grad_value=None,
            clip_grad_norm=None,
            param_group_modifiers=None,
            exclude_bias_from_wd=True,
            exclude_norm_from_wd=True,
            # info for schedules (only required when a schedule is used)
            end_checkpoint=None,
            updates_per_epoch=None,
            effective_batch_size=None,
    ):
        self.logger = logging.getLogger(type(self).__name__)
        self.model = model
        assert "lr" in torch_optim_ctor.keywords
        lr = torch_optim_ctor.keywords["lr"]
        self.lr_scaler = create(lr_scaler, lr_scaler_from_kwargs, base_lr=lr)
        if lr_scaler is not None:
            # can't use effective_batch_size as lr_scaler_factor because initializing schedules requires the
            # original effective_batch_size without adjustments to n_views
            # lr_scaler_factor == effective_batch_size if single view
            # lr_scaler_factor == n_views * effective_batch_size if multi-view
            new_lr = self.lr_scaler.get_scaled_lr(lr_scaler_factor)
            self.logger.info(f"unscaled lr: {float_to_scientific_notation(lr, max_precision=2)}")
            self.logger.info(f"scaled lr: {new_lr} ({self.lr_scaler} lr_scaler_factor={lr_scaler_factor})")
            torch_optim_ctor.keywords["lr"] = new_lr
        self.clip_grad_value = clip_grad_value
        self.clip_grad_norm = clip_grad_norm
        assert self.clip_grad_value is None or self.clip_grad_value > 0
        assert self.clip_grad_norm is None or self.clip_grad_norm > 0

        # create a param group for each parameter
        param_group_modifiers = create_collection(param_group_modifiers, param_group_modifier_from_kwargs)
        param_groups = []
        excl_wd_str = f"exclude_bias_from_wd={exclude_bias_from_wd} exclude_norm_from_wd={exclude_norm_from_wd}"
        self.logger.info(f"group modifiers {excl_wd_str} [{' '.join(str(pgm) for pgm in param_group_modifiers)}]")
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            properties = {}
            # excluding norm and bias params is very common for all models -> support with simple flag
            # bias has ndim == 1, so it needs to be checked before
            # the bias of norm layers is considered a bias, not a norm parameter
            if name.endswith(".bias") and exclude_bias_from_wd:
                properties["weight_decay"] = 0.
            # timm does it like this...not sure if other parameters can also have ndim <= 1
            # https://github.com/rwightman/pytorch-image-models/blob/master/timm/optim/optim_factory.py
            elif param.ndim <= 1 and exclude_norm_from_wd:
                properties["weight_decay"] = 0.

            for param_group_modifier in param_group_modifiers:
                for key, value in param_group_modifier.get_properties(model, name, param).items():
                    assert key not in properties
                    properties[key] = value
            assert "param" not in properties
            assert "name" not in properties
            properties["name"] = name
            properties["params"] = [param]
            param_groups.append(properties)

        # merge same groups with same parameters (useful for logging)
        merged_groups = []
        merged_groups_properties = []
        merged_groups_paramnames = []
        for param_group in param_groups:
            param_name = param_group.pop("name")
            properties = {k: v for k, v in param_group.items() if k != "params"}
            matching_group_idx = None
            for i, merged_group_properties in enumerate(merged_groups_properties):
                if properties == merged_group_properties:
                    matching_group_idx = i
                    break
            if matching_group_idx is None:
                merged_groups.append(param_group)
                merged_groups_properties.append(properties)
                merged_groups_paramnames.append([param_name])
            else:
                merged_groups[matching_group_idx]["params"] += param_group["params"]
                merged_groups_paramnames[matching_group_idx].append(param_name)

        # add name to param_groups
        for param_group in merged_groups:
            names = []
            for key, value in param_group.items():
                if key == "params":
                    continue
                if isinstance(value, float):
                    value_str = float_to_scientific_notation(value, max_precision=1, remove_plus=True)
                else:
                    raise NotImplementedError
                names.append(f"{key}={value_str}")
            if len(names) > 0:
                param_group["name"] = "&".join(names)

        # torch optimizer organizes parameters by enumerating them (not by name)
        # so for loading an arbitrary optim state_dict an association from param_name to param_idx has to be stored
        self.param_idx_to_name = Bidict()
        idx = 0
        for group_paramnames in merged_groups_paramnames:
            for param_name in group_paramnames:
                self.param_idx_to_name.set_forward(idx, param_name)
                idx += 1

        # initialize torch optim
        self.torch_optim = torch_optim_ctor(merged_groups)

        # for grad clipping all parameters of the optimizer are required
        self.all_parameters = None
        if self.clip_grad_value is not None or self.clip_grad_norm is not None:
            self.all_parameters = list(model.parameters())

        # scale lr (e.g. layerwise_lr_decay_modifier)
        for param_group in self.torch_optim.param_groups:
            if "lr_scale" in param_group:
                assert "original_lr" not in param_group
                param_group["original_lr"] = param_group["lr"]
                # lr is float so inplace operation is fine
                # this scaling is only relevant for logging and epoch based schedules
                # for update based schedule the value is anyway scaled again at the start of the update
                param_group["lr"] *= param_group["lr_scale"]
                self.logger.info(
                    f"scaled lr of param_group '{param_group['name']}' "
                    f"from {float_to_scientific_notation(param_group['original_lr'], max_precision=2)} "
                    f"to {float_to_scientific_notation(param_group['lr'], max_precision=2)}"
                )

        # create schedules
        self.schedule = schedule_from_schedule_configs(
            schedule,
            end_checkpoint=end_checkpoint,
            effective_batch_size=effective_batch_size,
            updates_per_epoch=updates_per_epoch,
            abs_max_value=self.torch_optim.defaults["lr"],
        )
        self.weight_decay_schedule = schedule_from_schedule_configs(
            weight_decay_schedule,
            end_checkpoint=end_checkpoint,
            effective_batch_size=effective_batch_size,
            updates_per_epoch=updates_per_epoch,
            abs_max_value=self.torch_optim.defaults["weight_decay"],
        )
        # store initial_lr/initial_wd in param_groups
        # NOTE: torch optimizer broadcasts all values to all param groups (so every param_group has a weight_decay)
        # LEGACY: "initial_lr" is not needed anymore with new schedules
        if self.schedule is not None:
            for param_group in self.torch_optim.param_groups:
                assert "initial_lr" not in param_group
                param_group["initial_lr"] = param_group["lr"]
        if self.weight_decay_schedule is not None:
            for param_group in self.torch_optim.param_groups:
                assert "initial_wd" not in param_group
                param_group["initial_wd"] = param_group["weight_decay"]

    def step(self, grad_scaler):
        # NOTE: closure is not supported with GradScaler
        if self.clip_grad_value is not None or self.clip_grad_norm is not None:
            grad_scaler.unscale_(self.torch_optim)
        # clip gradients
        if self.clip_grad_value is not None:
            torch.nn.utils.clip_grad_value_(self.all_parameters, self.clip_grad_value)
        if self.clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.all_parameters, self.clip_grad_norm)
        # torch optim step with grad scaler
        grad_scaler.step(self.torch_optim)
        grad_scaler.update()
        # after step (e.g. for EMA)
        self.model.after_update_step()

    def schedule_epoch_step(self, checkpoint):
        """ epoch is int """
        if self.schedule is not None and self.schedule.should_step_before_epoch(checkpoint):
            self._schedule_step(checkpoint)
        if self.weight_decay_schedule is not None and self.weight_decay_schedule.should_step_before_epoch(checkpoint):
            self._wd_schedule_step(checkpoint)

    def schedule_update_step(self, checkpoint):
        if self.schedule is not None and self.schedule.should_step_before_update(checkpoint):
            self._schedule_step(checkpoint)
        if self.weight_decay_schedule is not None and self.weight_decay_schedule.should_step_before_update(checkpoint):
            self._wd_schedule_step(checkpoint)

    def _schedule_step(self, checkpoint):
        lr_scale = self.schedule.get_value(checkpoint)
        is_new_schedule = self.schedule.get_is_new_schedule(checkpoint)
        for param_group in self.torch_optim.param_groups:
            # LEGACY
            if is_new_schedule:
                if "lr_scale" in param_group:
                    # scale the lr (scaled by the current progress from the schedule) with the scale from the
                    # parameter group (e.g. for layerwise-lr-decay)
                    param_group["lr"] = param_group["lr_scale"] * lr_scale
                else:
                    param_group["lr"] = lr_scale
            else:
                param_group["lr"] = param_group["initial_lr"] * lr_scale

    def _wd_schedule_step(self, checkpoint):
        wd_scale = self.weight_decay_schedule.get_value(checkpoint)
        for param_group in self.torch_optim.param_groups:
            param_group["weight_decay"] = param_group["initial_wd"] * wd_scale

    def zero_grad(self, set_to_none=True):
        # set_to_none is True by default (unlike torch.optim.optimizer)
        # because it has better performance (https://www.youtube.com/watch?v=9mS1fIYj1So)
        self.torch_optim.zero_grad(set_to_none)

    def state_dict(self):
        sd = self.torch_optim.state_dict()
        sd["param_idx_to_name"] = self.param_idx_to_name.to_forward()
        return sd

    def load_state_dict(self, state_dict_to_load):
        # state_dict doesn't have to have the same param_groups as the current state_dict
        # - freeze/unfreeze parameters
        # - add new parameters
        # - change weight_decay
        if "param_idx_to_name" in state_dict_to_load:
            # torch optim stores:
            # - a list of param_idxs in each param_group
            # - a dict from param_idxs to state for the state of the param
            # -> match the param_idxs and overwrite the state
            loaded_param_idx_to_name = Bidict(forward=state_dict_to_load["param_idx_to_name"])
            loaded_states = state_dict_to_load["state"]
            cur_state_dict = self.torch_optim.state_dict()
            cur_states = cur_state_dict["state"]
            cur_param_groups = cur_state_dict["param_groups"]
            # NOTE: iterating over cur_param_groups makes it such that parameters that are frozen in cur_param_groups
            # are not added to the state_dict (if one would iterate over the loaded_param_groups they would)
            for cur_param_group in cur_param_groups:
                for cur_param_idx in cur_param_group["params"]:
                    param_name = self.param_idx_to_name.get_forward(cur_param_idx)
                    loaded_param_idx = loaded_param_idx_to_name.get_backward(param_name)
                    if loaded_param_idx not in loaded_states:
                        self.logger.info(f"couldn't find a state to load for param {param_name}")
                        continue
                    # overwrite state with loaded state
                    cur_states[cur_param_idx] = loaded_states[loaded_param_idx]
            state_dict_to_load = dict(state=cur_states, param_groups=cur_param_groups)
        self.torch_optim.load_state_dict(state_dict_to_load)
