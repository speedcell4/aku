import inspect
import typing
from argparse import SUPPRESS
from itertools import zip_longest
from typing import Callable


def is_union(retype) -> bool:
    return getattr(retype, '__origin__', None) is typing.Union


def is_optional(retype) -> bool:
    return is_union(retype) and type(None) in getattr(retype, '__args__', [])


def get_annotations(func: Callable, only_with_default: bool = False):
    annotations = typing.get_type_hints(func)
    spec = inspect.getfullargspec(func)
    args = spec.args
    defaults = spec.defaults or []

    if only_with_default:
        return [(name, annotations.get(name, str), default, f'{name}')
                for name, default in zip(args[::-1], defaults[::-1])
                ][::-1]
    else:
        return [(name, annotations.get(name, str), default, f'{name}')
                for name, default in zip_longest(args[::-1], defaults[::-1], fillvalue=SUPPRESS)
                ][::-1]
