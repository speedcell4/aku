from typing import get_type_hints, Type, Callable, Any, Dict

ParsingFn = Callable[[str], Any]
parsing_fn_registry: Dict[Any, ParsingFn] = {}


def register_parsing_fn(fn: ParsingFn) -> ParsingFn:
    retype = get_type_hints(fn)['return']
    assert retype not in parsing_fn_registry, \
        f'the parsing function of {retype} is already registered ({parsing_fn_registry[retype]})'

    parsing_fn_registry[retype] = fn
    return fn


def get_parsing_fn(retype: Type) -> ParsingFn:
    return parsing_fn_registry.get(retype, retype)


@register_parsing_fn
def str2bool(option_string: str) -> bool:
    option_string = option_string.strip().lower()
    if option_string in ('1', 't', 'true', 'y', 'yes'):
        return True
    if option_string in ('0', 'f', 'false', 'n', 'no'):
        return False
    raise ValueError(f'{option_string} is not a boolean value.')


@register_parsing_fn
def str2none(option_string: str) -> type(None):
    option_string = option_string.strip().lower()
    if option_string in ('nil', 'null', 'none'):
        return True
    raise ValueError(f'{option_string} is not a null value.')
