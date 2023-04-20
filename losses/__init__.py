from utils.factory import instantiate


def loss_fn_from_kwargs(kind, update_counter=None, **kwargs):
    return instantiate(
        module_names=[f"losses.{kind}", "torch.nn"],
        type_names=[kind],
        # pass update_counter to SchedulableLoss but not to e.g. torch.nn.MSELoss
        optional_kwargs=dict(update_counter=update_counter),
        **kwargs,
    )
