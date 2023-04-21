import torch
from torch.nn.functional import normalize

from losses.nnclr_loss import nnclr_loss_fn
from utils.model_utils import update_ema, copy_params
from .nnclr_head import NnclrHead


class NnclrEmaOracleHead(NnclrHead):
    def __init__(self, target_factor=0.99, oracle_p=1., **kwargs):
        self.target_projector = None
        super().__init__(**kwargs)
        self.target_factor = target_factor
        self.oracle_p = oracle_p

    def load_state_dict(self, state_dict, strict=True):
        # patch for stage2
        if "target_projector" not in state_dict:
            for key in list(state_dict.keys()):
                if key.startswith("projector."):
                    state_dict[f"target_projector.{key[len('projector.'):]}"] = state_dict[key]
        super().load_state_dict(state_dict=state_dict, strict=strict)

    def _model_specific_initialization(self):
        copy_params(self.projector, self.target_projector)

    def register_components(self, input_dim, output_dim, **kwargs):
        super().register_components(input_dim=input_dim, output_dim=output_dim)
        self.target_projector = self.create_projector(input_dim, self.proj_hidden_dim, output_dim)
        for param in self.target_projector.parameters():
            param.requires_grad = False

    def _forward(self, pooled):
        target_projected = self.target_projector(pooled)
        projected = self.projector(pooled)
        predicted = self.predictor(projected)
        return dict(projected=target_projected, predicted=predicted)

    def after_update_step(self):
        update_ema(self.projector, self.target_projector, self.target_factor)

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
        loss0 = nnclr_loss_fn(predicted0, nn1, temperature=self.temperature, transposed=self.transposed)
        loss1 = nnclr_loss_fn(predicted1, nn0, temperature=self.temperature, transposed=self.transposed)
        loss = (loss0 + loss1) / 2

        nn_acc = self.calculate_nn_accuracy(normed_projected0, ids=idx, y=y, idx0=idx0_orig, nn0=nn0_orig)
        return loss, dict(nn_accuracy=nn_acc)

    @torch.no_grad()
    def find_nn(self, normed_projected, ids, y, retrieve_idx_orig=False):
        similarity_matrix = self.get_queue_similarity_matrix(normed_projected, ids=ids)
        ## option to pass original nn to  calculate nn-acc
        if retrieve_idx_orig:
            idx_original = similarity_matrix.max(dim=1)[1]
            nearest_neighbor_original = self.queue[idx_original]
        oracle_end_idx = int(len(similarity_matrix) * self.oracle_p)
        is_same_class = y[:oracle_end_idx, None] == self.queue_y[None, :]
        similarity_matrix[:oracle_end_idx][is_same_class] += 2.
        if self.topk == 0:
            idx = similarity_matrix.max(dim=1)[1]
        else:
            n = similarity_matrix.shape[0]
            candidate_idx = similarity_matrix.topk(self.topk, dim=1)[1]
            dice = torch.randint(size=(n,), high=self.topk)
            idx = candidate_idx[torch.arange(n), dice]
        # idx = similarity_matrix.max(dim=1)[1]
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
