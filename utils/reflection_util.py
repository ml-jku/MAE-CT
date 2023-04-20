import inspect


def get_all_ctor_kwarg_names(cls):
    result = set()
    _get_all_ctor_kwarg_names(cls, result)
    return result


def _get_all_ctor_kwarg_names(cls, result):
    for name in inspect.signature(cls).parameters.keys():
        result.add(name)
    if cls.__base__ is not None:
        _get_all_ctor_kwarg_names(cls.__base__, result)
