from kappadata import SubsetWrapper
# TODO import from kappadata
from kappadata.wrappers.dataset_wrappers.classwise_subset_wrapper import ClasswiseSubsetWrapper

def get_subset_identifier(dataset):
    result = None
    for wrapper in dataset.all_wrappers:
        if isinstance(wrapper, SubsetWrapper):
            # TODO better description which subset is used
            result = (result or "") + f".subset(size={len(wrapper.indices)})"
        if isinstance(wrapper, ClasswiseSubsetWrapper):
            # TODO better description which subset is used
            result = (result or "") + f".subset(size={len(wrapper.indices)})"
    return result or ""
