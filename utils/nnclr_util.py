import torch
import torch.nn.functional as F

from distributed.gather import all_gather_nograd


@torch.no_grad()
def find_nn(projected, ids, queue, queue_ids):
    similarity_matrix = get_queue_similarity_matrix(projected=projected, ids=ids, queue=queue, queue_ids=queue_ids)
    idx = similarity_matrix.max(dim=1)[1]
    nn = queue[idx]
    return idx, nn


@torch.no_grad()
def get_queue_similarity_matrix(projected, ids, queue, queue_ids):
    projected = F.normalize(projected, dim=-1)
    similarity_matrix = projected @ queue.T

    # check if queue contains embeddings of the same sample of the previous epoch
    is_own_id = queue_ids[None, :] == ids[:, None]
    # set similarity to self to -1
    similarity_matrix[is_own_id] = -1.

    return similarity_matrix


@torch.no_grad()
def update_queue(ids, projected, classes, queue_ids, queue, queue_classes, queue_ptr):
    ids = all_gather_nograd(ids)
    projected = all_gather_nograd(F.normalize(projected, dim=-1))
    classes = all_gather_nograd(classes)

    batch_size = projected.shape[0]
    queue_size = queue.shape[0]
    assert queue_size % batch_size == 0

    ptr_from = int(queue_ptr)
    ptr_to = ptr_from + batch_size
    queue_ids[ptr_from:ptr_to] = ids
    queue[ptr_from:ptr_to] = projected
    queue_classes[ptr_from:ptr_to] = classes
    queue_ptr[0] = ptr_to % queue_size
