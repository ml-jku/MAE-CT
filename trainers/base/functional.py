import logging

import einops
import torch
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader

from utils.functional import get_powers_of_two, is_power_of_two
from .noop_context import NoopContext
from .noop_grad_scaler import NoopGradScaler


def calculate_effective_batch_size_per_device(effective_batch_size, world_size):
    msg = "effective_batch_size needs to be multiple of world_size"
    assert effective_batch_size % world_size == 0, msg
    effective_batch_size_per_device = int(effective_batch_size / world_size)
    return effective_batch_size_per_device


def calculate_batch_size_and_accumulation_steps(effective_batch_size_per_device, max_batch_size=None):
    # calculate batch_size and accumulation_steps
    if max_batch_size is None:
        batch_size = effective_batch_size_per_device
        accumulation_steps = 1
    else:
        if effective_batch_size_per_device <= max_batch_size:
            # fits into memory
            batch_size = effective_batch_size_per_device
            accumulation_steps = 1
        else:
            # multiple accumulation steps
            msg = "effective_batch_size_per_device needs to be multiple of max_batch_size"
            assert effective_batch_size_per_device % max_batch_size == 0, msg
            accumulation_steps = int(effective_batch_size_per_device / max_batch_size)
            batch_size = int(effective_batch_size_per_device / accumulation_steps)
    return batch_size, accumulation_steps


@torch.no_grad()
def backup_all_buffers(model):
    buffers = {}
    for name, buffer in model.named_buffers():
        buffers[name] = buffer.clone()
    return buffers


@torch.no_grad()
def restore_all_buffers(model, buffers):
    for name, buffer in model.named_buffers():
        buffer.data.copy_(buffers[name])
    return buffers


def calculate_automatic_max_batch_size(train_dataset, train_step_fn, effective_batch_size_per_device, device, model):
    if str(device) == "cpu":
        return effective_batch_size_per_device
    # batchsizes that are not a power of two are not supported
    if not is_power_of_two(effective_batch_size_per_device):
        return effective_batch_size_per_device
    # restore buffers as some are modified by an update (e.g. DINO center buffer)
    buffer_bkp = backup_all_buffers(model)

    # compose batch_sizes to try (start from 2 because some models do batchnorm during training [e.g. barlow twins])
    batch_sizes = get_powers_of_two(2, effective_batch_size_per_device)

    # make a train_step with decreasing batch_sizes (faster when batchsize is actually correct)
    sample, ctx = next(iter(DataLoader(train_dataset, batch_size=1)))
    max_batch_size = 1
    for batch_size in reversed(batch_sizes):
        logging.info(f"trying batch_size {batch_size}")

        # scale batch_size by repeating the sample
        if isinstance(sample, (list, tuple)):
            data = []
            for item in sample:
                if isinstance(item, (list, tuple)):
                    data.append([einops.repeat(entry, "1 ... -> bs ...", bs=batch_size) for entry in item])
                else:
                    data.append(einops.repeat(item, "1 ... -> bs ...", bs=batch_size))
        else:
            data = einops.repeat(sample, "1 ... -> bs ...", bs=batch_size)
        # wrap into tuple
        if isinstance(data, list):
            data = tuple(data)
        # scale batch_size of ctx
        ctx = {
            k: einops.repeat(v, "1 ... -> bs ...", bs=batch_size) if torch.is_tensor(v) else v
            for k, v in ctx.items()
        }

        # try update step
        try:
            train_step_fn((data, ctx), train_dataset=train_dataset)
            max_batch_size = batch_size
            break
        except RuntimeError as e:
            if not str(e).startswith("CUDA out of memory"):
                raise e

    # restore buffers
    restore_all_buffers(model, buffer_bkp)
    return max_batch_size


def get_grad_scaler_and_autocast_context(precision, device):
    if precision == torch.float32:
        return NoopGradScaler(), NoopContext()
    if precision == torch.bfloat16:
        # GradScaler shouldn't be necessary (https://github.com/pytorch/pytorch/issues/36169)
        return NoopGradScaler(), torch.autocast(str(device), dtype=precision)
    elif precision == torch.float16:
        return GradScaler(), torch.autocast(str(device), dtype=precision)
    raise NotImplementedError
