from collections import defaultdict
from functools import partial

from distributed.gather import all_gather_nograd
from loggers.base.dataset_logger import DatasetLogger
from utils.object_from_kwargs import objects_from_kwargs
from utils.running_mean import RunningMean


class LossLogger(DatasetLogger):
    def __init__(self, forward_kwargs=None, **kwargs):
        super().__init__(**kwargs)
        self.forward_kwargs = objects_from_kwargs(forward_kwargs)

    def _forward_loss(self, batch, model, trainer):
        with trainer.autocast_context:
            outputs = trainer.forward(model, batch, self.dataset, **self.forward_kwargs)
            losses, _ = trainer.get_loss(outputs, model)
        batch_size = len(batch[0][0])
        return {loss_name: loss.cpu() for loss_name, loss in losses.items()}, batch_size

    def get_dataset_mode(self, trainer):
        return trainer.dataset_mode

    @property
    def return_ctx(self):
        return True

    # noinspection PyMethodOverriding
    def _log(self, update_counter, model, trainer, logger_info_dict, **_):
        forward_results = self.iterate_over_dataset(
            forward_fn=partial(self._forward_loss, model=model, trainer=trainer),
            update_counter=update_counter,
        )
        # TODO inaccurate with DDP
        loss_rms = defaultdict(RunningMean)
        for forward_result, batch_size in forward_results:
            for key, value in forward_result.items():
                loss_rms[key].update(value, count=batch_size)
        loss_means = {key: value.mean for key, value in loss_rms.items()}
        loss_counts = {key: value.count for key, value in loss_rms.items()}

        # gather
        gathered_loss_means = {k: all_gather_nograd(v) for k, v in loss_means.items()}
        gathered_loss_counts = {k: all_gather_nograd(v) for k, v in loss_counts.items()}

        # running mean over gathered results
        loss_rms = defaultdict(RunningMean)
        for key, g_mean in gathered_loss_means.items():
            g_count = gathered_loss_counts[key]
            for i in range(len(g_mean)):
                loss_rms[key].update(g_mean[i], count=g_count[i])

        # extract mean from gathered running mean
        final_losses = {k: v.mean for k, v in loss_rms.items()}

        # log
        for loss_name, loss in final_losses.items():
            key = f"loss/{self.dataset_key}/{loss_name}"
            self.logger.info(f"{key}: {loss:.5f}")
            logger_info_dict[key] = loss.item()
            self.writer.add_scalar(key, loss, update_counter=update_counter)
