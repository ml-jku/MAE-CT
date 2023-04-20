from torch.nn.functional import normalize

from losses.nnclr_loss import nnclr_loss_fn
from .base.contrastive_head_base import ContrastiveHeadBase


class NnclrNoqueueHead(ContrastiveHeadBase):
    def __init__(self, temperature, proj_hidden_dim, pred_hidden_dim,transposed=False, **kwargs):
        self.proj_hidden_dim = proj_hidden_dim
        self.pred_hidden_dim = pred_hidden_dim
        self.projector, self.predictor = None, None
        super().__init__(**kwargs)
        self.temperature = temperature
        self.transposed = transposed

    def register_components(self, input_dim, output_dim, **kwargs):
        self.projector = self.create_projector(input_dim, self.proj_hidden_dim, output_dim)
        self.predictor = self.create_predictor(output_dim, self.pred_hidden_dim)

    def _forward(self, pooled):
        projected = self.projector(pooled)
        predicted = self.predictor(projected)
        return dict(projected=projected, predicted=predicted)

    def _get_loss(self, outputs, idx, y):
        projected0 = outputs["view0"]["projected"]
        projected1 = outputs["view1"]["projected"]
        predicted0 = outputs["view0"]["predicted"]
        predicted1 = outputs["view1"]["predicted"]

        normed_projected0 = normalize(projected0, dim=-1)
        normed_projected1 = normalize(projected1, dim=-1)

        loss0 = nnclr_loss_fn(predicted0, normed_projected1, temperature=self.temperature, transposed=self.transposed)
        loss1 = nnclr_loss_fn(predicted1, normed_projected0, temperature=self.temperature, transposed=self.transposed)
        loss = (loss0 + loss1) / 2

        nn_acc = self.calculate_nn_accuracy(normed_projected0, ids=idx, y=y)
        return loss, dict(nn_accuracy=nn_acc)
