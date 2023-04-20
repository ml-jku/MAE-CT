import torch
from torch.nn.functional import normalize

from losses.nnclr_loss import nnclr_loss_fn
from .nnclr_noqueue_head import NnclrNoqueueHead


class NnclrOracleHead(NnclrNoqueueHead):
    def __init__(self, local_scaling_knn=0, **kwargs):
        super().__init__(**kwargs)
        self.local_scaling_knn = local_scaling_knn

    def _get_loss(self, outputs, idx, y):
        projected0 = outputs["view0"]["projected"]
        projected1 = outputs["view1"]["projected"]
        predicted0 = outputs["view0"]["predicted"]
        predicted1 = outputs["view1"]["predicted"]

        normed_projected0 = normalize(projected0, dim=-1)
        normed_projected1 = normalize(projected1, dim=-1)

        # find nn
        idx0, nn0, idx0_orig, nn0_orig = self.find_nn(
            normed_projected0, ids=idx, y=y, retrieve_idx_orig=True)
        _, nn1 = self.find_nn(normed_projected1, ids=idx, y=y)

        # nn is calculated with no_grad so gradient only flows back through 'predicted'
        loss0 = nnclr_loss_fn(predicted0, nn1, temperature=self.temperature)
        loss1 = nnclr_loss_fn(predicted1, nn0, temperature=self.temperature)
        loss = (loss0 + loss1) / 2

        nn_acc = self.calculate_nn_accuracy(normed_projected0, ids=idx, y=y, idx0=idx0_orig, nn0=nn0_orig)
        return loss, dict(nn_accuracy=nn_acc)


    @torch.no_grad()
    def find_nn(self, normed_projected, ids, y, retrieve_idx_orig=False):
        similarity_matrix = self.get_queue_similarity_matrix(normed_projected, ids=ids)
        is_same_class = y[:, None] == self.queue_y[None, :]
        ## option to pass original nn to  calculate nn-acc
        if retrieve_idx_orig:
            idx_original = similarity_matrix.max(dim=1)[1]
            nearest_neighbor_original = self.queue[idx_original]
        similarity_matrix[is_same_class] += 2.
        idx = similarity_matrix.max(dim=1)[1]
        nearest_neighbor = self.queue[idx]
        if retrieve_idx_orig:
            return idx, nearest_neighbor, idx_original, nearest_neighbor_original
        else:
            return idx, nearest_neighbor

    @torch.no_grad()
    def get_queue_similarity_matrix(self, normed_projected, ids):
        similarity_matrix = normed_projected @ self.queue.T
        if self.exclude_self_from_queue:
            # check if queue contains embeddings of the same sample of the previous epoch
            is_own_id = self.queue_id[None, :] == ids[:, None]
            # set similarity to self to -1
            similarity_matrix[is_own_id] = -1.
        return similarity_matrix