from loggers.base.logger_base import LoggerBase
from models.base.composite_model_base import CompositeModelBase
from utils.model_utils import get_named_models


class LrLogger(LoggerBase):
    def should_log_after_update(self, checkpoint):
        if checkpoint.update == 1:
            return True
        return super().should_log_after_update(checkpoint)

    # noinspection PyMethodOverriding
    def _log(self, update_counter, interval_type, model, **_):
        for model_name, model in get_named_models(model).items():
            if isinstance(model, CompositeModelBase) or model.optim is None:
                continue

            for param_group in model.optim.torch_optim.param_groups:
                group_name = f"/{param_group['name']}" if "name" in param_group else ""
                if model.optim.schedule is not None:
                    lr = param_group["lr"]
                    self.writer.add_scalar(
                        key=f"optim/lr/{model_name}{group_name}",
                        value=lr,
                        update_counter=update_counter,
                    )
                    # self.logger.info(f"optim/lr/{model_name}{group_name}: {lr}")
                if model.optim.weight_decay_schedule is not None:
                    wd = param_group["weight_decay"]
                    self.writer.add_scalar(
                        key=f"optim/wd/{model_name}{group_name}",
                        value=wd,
                        update_counter=update_counter,
                    )
                    # self.logger.info(f"optim/wd/{model_name}{group_name}: {wd}")
