from collections import defaultdict
import torch
import math
import torch.nn.functional as F
import einops

# TODO split multiclass/binary into seperate methods
# TODO automatic batchsize
@torch.no_grad()
def knn_metrics(
        train_x,
        test_x,
        train_y,
        test_y,
        knns,
        batch_normalize=False,
        batch_size=None,
        weights="distance",
        eps: float = 1e-5,  # default value of torch.nn.BatchNorm1d.eps
):
    # check x and y
    assert len(train_x) == len(train_y) and len(test_x) == len(test_y)
    assert train_x.ndim == 2 and train_y.ndim == 1 and test_x.ndim == 2 and test_y.ndim == 1
    # check knns
    assert isinstance(knns, (tuple, list))
    assert all(isinstance(knn, int) and knn >= 1 for knn in knns)
    # check batch_size
    assert batch_size is None or (batch_size is not None and isinstance(batch_size, int) and batch_size >= 1)
    # check weights
    assert weights in ["uniform", "distance"]

    # filter knns that are larger than number of train samples
    knns = [knn for knn in knns if knn <= len(train_y)]

    # apply batch normalization
    if batch_normalize:
        mean = train_x.mean(dim=0)
        std = train_x.std(dim=0) + eps
        train_x = (train_x - mean) / std
        test_x = (test_x - mean) / std

    # normalize to length 1 for cosine distance
    train_x = F.normalize(train_x, dim=1)
    test_x = F.normalize(test_x, dim=1)

    # return purity/accuracy per knn
    purities = defaultdict(lambda: torch.tensor(0, device=train_x.device))
    accuracies = defaultdict(lambda: torch.tensor(0, device=train_x.device))
    scores = defaultdict(list)
    # initialize onehot vector per class (used for counting votes in classification)
    n_classes = max(train_y.max().item(), test_y.max().item()) + 1
    class_onehot = torch.diag(torch.ones(max(2, n_classes), device=train_x.device))
    # calculate in chunks to avoid OOM
    n_chunks = math.ceil(len(test_y) / (batch_size or len(test_y)))


    for test_x_chunk, test_y_chunk in zip(test_x.chunk(n_chunks), test_y.chunk(n_chunks)):
        # the purity comparison (test_y_chunk == nn_labels) actually works without this but do this anyways
        # to avoid weird errors with shape broadcasting
        test_y_chunk_purity = einops.rearrange(test_y_chunk, "n_test -> 1 n_test")
        # calculate similarity
        similarities = test_x_chunk @ train_x.T
        for knn in knns:
            # retrieve k-nearest-neighbors and their labels
            # in some cases it might be faster if this would be outside the knn loop with sorted=True and knn=max(knns)
            # the trade-off is (not sure how large len(knns) has to be for the current version to be slower):
            # - large len(knns): sorted=True + knn=max(knns) + "topk_indices = ..." outside of knns loop
            # - small len(knns): current version
            topk_similarities, topk_indices = similarities.topk(k=knn, dim=1)
            flat_topk_indices = einops.rearrange(topk_indices, "n_test knn -> (n_test knn)")
            flat_nn_labels = train_y[flat_topk_indices]

            # calculate accuracy of a knn classifier
            flat_nn_onehot = class_onehot[flat_nn_labels]
            nn_onehot = einops.rearrange(flat_nn_onehot, "(n_test knn) n_classes -> knn n_test n_classes", knn=knn)
            if weights == "uniform":
                logits = nn_onehot.sum(dim=0)
                knn_classes = logits.argmax(dim=1)
            elif weights == "distance":
                # 0.07 is used as default by DINO/solo-learn
                # https://github.com/facebookresearch/dino/blob/main/eval_knn.py#L196
                # https://github.com/vturrisi/solo-learn/blob/main/solo/utils/knn.py#L31
                topk_similarities = topk_similarities.div_(0.07).exp_()
                topk_similarities = einops.rearrange(topk_similarities, "n_test knn -> knn n_test 1")
                logits = (nn_onehot * topk_similarities).sum(dim=0)
                knn_classes = logits.argmax(dim=1)
            else:
                raise NotImplementedError

            accuracies[knn] += (test_y_chunk == knn_classes).sum()

            # calculate purity
            nn_labels = einops.rearrange(flat_nn_labels, "(knn n_test) -> knn n_test", knn=knn)
            purities[knn] += (test_y_chunk_purity == nn_labels).sum()

            # calculate score for binary classification
            if n_classes <= 2:
                score = logits[:, 1] - logits[:, 0]
                scores[knn].append(score)


    # counts to percent (and convert to primitive type)
    accuracies = {knn: (v / len(test_y)).item() for knn, v in accuracies.items()}
    purities = {knn: (v / (len(test_y) * knn)).item() for knn, v in purities.items()}
    if n_classes <= 2:
        scores = {key: torch.concat(score) for key, score in scores.items()}
    else:
        scores = None
    return accuracies, purities, scores