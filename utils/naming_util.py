def join_names(name1, name2):
    if name1 is None:
        return name2
    assert name2 is not None
    return f"{name1}.{name2}"


def camel_to_snake(camel_case_str):
    return ''.join(['_' + c.lower() if c.isupper() else c for c in camel_case_str]).lstrip('_')


def _type_name(obj, to_name_fn):
    from kappadata import KDSubset
    if isinstance(obj, KDSubset):
        return _type_name(obj.dataset, to_name_fn)
    cls = type(obj)
    return to_name_fn(cls.__name__)


def lower_type_name(obj):
    return _type_name(obj, lambda name: name.lower())


def snake_type_name(obj):
    return _type_name(obj, camel_to_snake)
