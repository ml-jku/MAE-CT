from utils.factory import instantiate


def dataset_from_kwargs(
        kind,
        dataset_config_provider,
        dataset_wrappers=None,
        sample_wrappers=None,
        **kwargs,
):
    dataset = instantiate(
        module_names=[f"datasets.{kind}"],
        type_names=[kind],
        dataset_config_provider=dataset_config_provider,
        **kwargs,
    )
    if dataset_wrappers is not None:
        assert isinstance(dataset_wrappers, list)
        for dataset_wrapper_kwargs in dataset_wrappers:
            dataset_wrapper_kind = dataset_wrapper_kwargs.pop("kind")
            dataset = instantiate(
                module_names=[f"kappadata.wrappers.dataset_wrappers.{dataset_wrapper_kind}"],
                type_names=[dataset_wrapper_kind],
                dataset=dataset,
                **dataset_wrapper_kwargs,
            )
    if sample_wrappers is not None:
        assert isinstance(sample_wrappers, list)
        for sample_wrapper_kwargs in sample_wrappers:
            sample_wrapper_kind = sample_wrapper_kwargs.pop("kind")
            dataset = instantiate(
                module_names=[
                    f"datasets.sample_wrappers.{sample_wrapper_kind}",
                    f"kappadata.wrappers.sample_wrappers.{sample_wrapper_kind}",
                ],
                type_names=[sample_wrapper_kind],
                dataset=dataset,
                **sample_wrapper_kwargs,
            )
    return dataset
