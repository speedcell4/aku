import argparse
import inspect
from typing import Any, Callable, Type

from aku import utils

_parsing_fn = {}


def register_parsing_fn(func: Callable[[str], Any]) -> Callable[[str], Any]:
    spec = inspect.getfullargspec(func)
    annotations = spec.annotations
    if inspect.isclass(func):
        _parsing_fn[func] = func
    elif 'return' in annotations:
        _parsing_fn[annotations['return']] = func
    else:
        raise KeyError(f'{func} is already registered')
    return func


def get_parsing_fn(retype: Type) -> Callable[[str], Any]:
    if utils.is_union(retype):
        def parsing_fn(option_string: str) -> retype:
            for fn in getattr(retype, '__args__', []):
                try:
                    return get_parsing_fn(fn)(option_string)
                except (ValueError, argparse.ArgumentTypeError):
                    pass
            raise argparse.ArgumentTypeError

        return parsing_fn

    return _parsing_fn.get(retype, retype)


@register_parsing_fn
def str2null(option_string: str) -> type(None):
    option_string = option_string.strip().lower()
    if option_string in ('nil', 'null', 'none'):
        return None
    raise argparse.ArgumentTypeError(f'"{option_string}" can not be parsed as null value')


@register_parsing_fn
def str2bool(option_string: str) -> bool:
    option_string = option_string.strip().lower()
    if option_string in ('1', 't', 'true', 'y', 'yes'):
        return True
    if option_string in ('0', 'f', 'false', 'n', 'no'):
        return False
    raise argparse.ArgumentTypeError(f'"{option_string}" can not be parsed as boolean value')
