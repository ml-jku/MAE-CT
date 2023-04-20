from loggers.base.logger_base import LoggerBase
from models.base.composite_model_base import CompositeModelBase
from utils.model_utils import get_named_models


class FreezerLogger(LoggerBase):
    def should_log_after_update(self, checkpoint):
        if checkpoint.update == 1:
            return True
        return super().should_log_after_update(checkpoint)

    # noinspection PyMethodOverriding
    def _log(self, update_counter, interval_type, model, **_):
        for model_name, model in get_named_models(model).items():
            if isinstance(model, CompositeModelBase):
                continue
            for freezer in model.freezers:
                if freezer.schedule is None:
                    continue

                self.writer.add_scalar(
                    key=f"freezers/{model_name}/{freezer}/is_frozen",
                    value=freezer.is_frozen,
                    update_counter=update_counter,
                )
