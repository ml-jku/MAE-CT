import logging

import torch

FLOAT32_ALIASES = ["float32", 32]
FLOAT16_ALIASES = ["float16", 16]
BFLOAT16_ALIASES = ["bfloat16", "bf16"]
VALID_PRECISIONS = FLOAT32_ALIASES + FLOAT16_ALIASES + BFLOAT16_ALIASES


def get_supported_precision(desired_precision, device):
    assert desired_precision in VALID_PRECISIONS
    if desired_precision in FLOAT32_ALIASES:
        return torch.float32
    if desired_precision in FLOAT16_ALIASES:
        desired_precision = "float16"
    if desired_precision in BFLOAT16_ALIASES:
        desired_precision = "bfloat16"

    if desired_precision == "bfloat16":
        if is_bfloat16_compatible(device):
            return torch.bfloat16
        else:
            # old cuda devices don't support bfloat16
            if is_float16_compatible(device):
                logging.info("bfloat16 not supported -> using float32 (float16 could lead to under-/overflows)")
                return torch.float32

    if desired_precision == "float16":
        if is_float16_compatible(device):
            return torch.float16
        else:
            # currently cpu only supports bfloat16
            if is_bfloat16_compatible(device):
                logging.info(f"float16 not supported -> using bfloat16")
                return torch.bfloat16

    logging.info(f"float16/bfloat16 not supported -> using float32")
    return torch.float32


def _is_compatible(device, dtype):
    try:
        with torch.autocast(device_type=str(device), dtype=dtype):
            pass
    except RuntimeError:
        return False
    return True


def is_bfloat16_compatible(device):
    return _is_compatible(device, torch.bfloat16)


def is_float16_compatible(device):
    return _is_compatible(device, torch.float16)
