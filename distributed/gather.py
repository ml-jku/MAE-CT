import torch
import torch.distributed as dist

from .config import is_distributed, get_world_size
from .functional.all_gather_grad_autograd import AllGatherGradAutograd
from .functional.all_gather_grad_overwrite import AllGatherGradOverwrite


def get_device_and_bfloat16supported():
    # gloo cpu -> okay
    # gloo cuda -> okay (although https://pytorch.org/docs/stable/distributed.html says it isn't supported)
    # nccl cpu -> fail (but gloo anyway recommended for cpu multiprocessing)
    # nccl cuda -> okay
    # bfloat16 cpu -> fail
    if not is_distributed():
        return torch.device("cpu"), True
    if dist.get_backend() == "nccl":
        return torch.device("cuda"), True
    if dist.get_backend() == "gloo":
        return torch.device("cpu"), False
    raise NotImplementedError


def _prepare_tensor(x):
    """
    prepare for distributed communication
    - wrap primitive types into tensors
    - push tensor onto supported device
    """
    device, bfloat16_supported = get_device_and_bfloat16supported()
    # I think this doesn't work in some configuration not sure in which though
    # note in which configuration and convert back to bool after gather
    if isinstance(x, bool):
        # x = torch.tensor(x, dtype=torch.float32, device=device)
        # og_device = torch.device("cpu")
        raise RuntimeError
    if isinstance(x, (float, int, list, tuple)):
        x = torch.tensor(x, device=device)
        og_device = torch.device("cpu")
    else:
        og_device = x.device
    if x.dtype == torch.bfloat16 and not bfloat16_supported:
        x = x.type(torch.float32)
    return x.to(device), og_device


def _all_gather_grad(x, all_gather_fn):
    x, og_device = _prepare_tensor(x)
    if is_distributed():
        result = all_gather_fn(x)
        if result[0].ndim == 0:
            # scalars can't be concatenated
            result = [r.unsqueeze(0) for r in result]
        return torch.concat(result).to(og_device)
    return _all_gather_nondistributed(x, og_device)


def all_gather_grad(x):
    return _all_gather_grad(x, AllGatherGradAutograd.apply)
    # return _all_gather_grad(x, AllGatherGradOverwrite.apply)


def all_gather_grad_autograd(x):
    return _all_gather_grad(x, AllGatherGradAutograd.apply)


def all_gather_grad_overwrite(x):
    return _all_gather_grad(x, AllGatherGradOverwrite.apply)


@torch.no_grad()
def all_gather_nograd(x):
    x, og_device = _prepare_tensor(x)
    if is_distributed():
        result = [torch.zeros_like(x) for _ in range(get_world_size())]
        dist.all_gather(result, x)
        if result[0].ndim == 0:
            # scalars can't be concatenated
            return torch.tensor(result, device=og_device)
        return torch.concat(result).to(og_device)
    return _all_gather_nondistributed(x, og_device).detach()


def _all_gather_nondistributed(x, og_device):
    if x.ndim == 0:
        # distributed gather adds a dimension to scalars
        x = x.unsqueeze(0)
    return x.to(og_device)


def all_gather_nograd_clipped(x, max_length):
    result = all_gather_nograd(x)
    if is_distributed():
        # DistributedSampler pads the dataset to give every GPU the same amount of samples
        return result[:max_length]
    return result


def all_reduce_sum_grad(x):
    x, og_device = _prepare_tensor(x)
    if is_distributed():
        # all_reduce is differentiable https://github.com/pytorch/pytorch/issues/58005
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
    return x.to(og_device)


def all_reduce_mean_grad(x):
    x, og_device = _prepare_tensor(x)
    if is_distributed():
        x = all_reduce_sum_grad(x) / get_world_size()
    return x.to(og_device)
