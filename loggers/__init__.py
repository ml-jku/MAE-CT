from utils.factory import instantiate


def logger_from_kwargs(kind, **kwargs):
    return instantiate(module_names=[
        f"loggers.{kind}",
        f"loggers.default_loggers.{kind}",
    ], type_names=[kind], **kwargs)
