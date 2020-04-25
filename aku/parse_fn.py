import inspect

registry = {}


def register_parse_fn(tp):
    ret = inspect.getfullargspec(tp).annotations['return']
    assert ret not in registry

    registry[ret] = tp
    return tp


def get_parse_fn(tp):
    return registry.get(tp, tp)


@register_parse_fn
def parse_bool(option_string: str) -> bool:
    option_string = option_string.strip().lower()
    if option_string in ('1', 'y', 'yes', 't', 'true'):
        return True
    if option_string in ('0', 'n', 'no', 'f', 'false'):
        return False
    raise ValueError(f'{option_string} is not a {bool.__name__} value')
