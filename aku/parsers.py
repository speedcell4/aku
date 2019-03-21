import inspect
from typing import Any, Callable, Type

from aku import utils

_parsing_fn = {}


def register_parsing_fn(func: Callable[[str], Any]) -> Callable[[str], Any]:
    spec = inspect.getfullargspec(func)
    annotations = spec.annotations
    if not inspect.isclass(func) and 'return' not in annotations:
        raise KeyError
    _parsing_fn[annotations['return']] = func
    return func


def get_parsing_fn(retype: Type) -> Callable[[str], Any]:
    if retype in _parsing_fn:
        return _parsing_fn[retype]

    if utils.is_union(retype):
        @register_parsing_fn
        def parsing_fn(option_string: str) -> retype:
            for fn in getattr(retype, '__args__', []):
                try:
                    return fn(option_string)
                except ValueError:
                    pass
            raise ValueError

        return parsing_fn

    raise ValueError


@register_parsing_fn
def str2null(option_string: str) -> None:
    option_string = option_string.strip().lower()
    if option_string in ('nil', 'null', 'none'):
        return None
    raise ValueError(f'"{option_string}" can not be parsed as null value')


@register_parsing_fn
def str2bool(option_string: str) -> bool:
    option_string = option_string.strip().lower()
    if option_string in ('1', 't', 'true', 'y', 'yes'):
        return True
    if option_string in ('0', 'f', 'false', 'n', 'no'):
        return False
    raise ValueError(f'"{option_string}" can not be parsed as boolean value')
