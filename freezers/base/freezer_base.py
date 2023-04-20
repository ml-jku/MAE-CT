import logging

from schedules import schedule_from_kwargs
from utils.update_counter import UpdateCounter


class FreezerBase:
    def __init__(self, update_counter: UpdateCounter, schedule=None):
        self.logger = logging.getLogger(type(self).__name__)
        self.update_counter = update_counter
        self.schedule = schedule_from_kwargs(schedule, update_counter=update_counter)
        self._requires_grad = None

        # check if children overwrite the correct method
        assert type(self).before_accumulation_step == FreezerBase.before_accumulation_step

    @property
    def is_frozen(self):
        return not self._requires_grad

    def __repr__(self):
        return str(self)

    def __str__(self):
        raise NotImplementedError

    def _change_state(self, model, requires_grad):
        raise NotImplementedError

    def after_weight_init(self, model):
        if self.schedule is None:
            self.logger.info(f"applying freezer {self} to {model.name}")
            self._change_state(model, requires_grad=False)
            self._requires_grad = False

    def before_accumulation_step(self, model):
        if self.schedule is not None:
            value = self.schedule.get_value(self.update_counter.cur_checkpoint)
            if value == 1:
                if self._requires_grad or self._requires_grad is None:
                    self.logger.info(f"change {model.name}.{self}.is_frozen to True")
                    self._requires_grad = False
            elif value == 0:
                if not self._requires_grad or self._requires_grad is None:
                    self.logger.info(f"change {model.name}.{self}.is_frozen to False")
                    self._requires_grad = True
            else:
                raise NotImplementedError
            self._change_state(model, requires_grad=self._requires_grad)
        self._before_accumulation_step(model)

    def _before_accumulation_step(self, model):
        """
        allow model to keep components in eval mode
        (model is set to train mode before the before_accumulation_step call)
        """
        raise NotImplementedError
