import collections.abc
from itertools import repeat
from pathlib import Path


# adapted from timm (timm/models/layers/helpers.py)
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            assert len(x) == n
            return x
        return tuple(repeat(x, n))

    return parse


def _is_ntuple(n):
    def check(x):
        return isinstance(x, tuple) and len(param) == n

    return check


to_2tuple = _ntuple(2)
is_2tuple = _is_ntuple(2)


def float_to_integer_exact(f):
    assert f.is_integer()
    return int(f)


def check_exclusive(*args):
    return sum(arg is not None for arg in args) == 1


def check_inclusive(*args):
    return sum(arg is not None for arg in args) in [0, len(args)]


def to_path(path):
    if path is not None and not isinstance(path, Path):
        return Path(path).expanduser()
    return path
