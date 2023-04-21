import torch
from torch.nn.functional import normalize

from losses.nnclr_loss import nnclr_loss_fn
from .nnclr_noqueue_head import NnclrNoqueueHead


class NnclrHead(NnclrNoqueueHead):
    def __init__(self, local_scaling_knn=0, topk=0, **kwargs):
        super().__init__(**kwargs)
        self.local_scaling_knn = local_scaling_knn
        self.topk = topk

    def _get_loss(self, outputs, idx, y):
        projected0 = outputs["view0"]["projected"]
        projected1 = outputs["view1"]["projected"]
        predicted0 = outputs["view0"]["predicted"]
        predicted1 = outputs["view1"]["predicted"]

        normed_projected0 = normalize(projected0, dim=-1)
        normed_projected1 = normalize(projected1, dim=-1)

        # find nn
        idx0, nn0 = self.find_nn(normed_projected0, ids=idx, topk=self.topk)
        _, nn1 = self.find_nn(normed_projected1, ids=idx, topk=self.topk)

        # nn is calculated with no_grad so gradient only flows back through 'predicted'
        loss0 = nnclr_loss_fn(predicted0, nn1, temperature=self.temperature, transposed=self.transposed)
        loss1 = nnclr_loss_fn(predicted1, nn0, temperature=self.temperature, transposed=self.transposed)
        loss = (loss0 + loss1) / 2

        nn_acc = self.calculate_nn_accuracy(normed_projected0, ids=idx, y=y, idx0=idx0, nn0=nn0)
        return loss, dict(nn_accuracy=nn_acc)

    @torch.no_grad()
    def get_queue_similarity_matrix(self, normed_projected, ids):
        similarity_matrix = super().get_queue_similarity_matrix(normed_projected, ids=ids)
        if self.local_scaling_knn == 0:
            return similarity_matrix

        # apply local scaling for hubness reduction
        distance_matrix = similarity_matrix
        # retrieve distances of k nearest neighbors
        nearest_neighbor_distances_z = distance_matrix.topk(dim=1, sorted=True, k=self.local_scaling_knn)[0]
        nearest_neighbor_distances_queue = distance_matrix.topk(dim=0, sorted=True, k=self.local_scaling_knn)[0]

        # take distance of furthest neighbor
        sigma_z = nearest_neighbor_distances_z[:, 0]
        sigma_queue = nearest_neighbor_distances_queue[0, :]

        # scale distances
        sigma_matrix = sigma_z[:, None] @ sigma_queue[None, :]
        return 1 - torch.exp(-(distance_matrix ** 2) / sigma_matrix)
