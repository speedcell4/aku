import inspect
import typing
from argparse import SUPPRESS
from itertools import zip_longest
from typing import Callable, Union

NoneType = type(None)


def is_union(retype) -> bool:
    return getattr(retype, '__origin__', None) is typing.Union


def unwrap_union(retype):
    return retype.__args__


def is_optional(retype) -> bool:
    return is_union(retype) and NoneType in getattr(retype, '__args__', [])


def unwrap_optional(retype):
    return Union[tuple(ty for ty in retype.__args__ if ty is not NoneType)]


def is_list(retype) -> bool:
    return getattr(retype, '_name', None) == 'List'


def unwrap_list(retype):
    return retype.__args__[0]


def is_homo_tuple(retype) -> bool:
    if getattr(retype, '_name', None) == 'Tuple':
        if retype.__args__.__len__() == 2 and ... in retype.__args__:
            return True
    return False


def unwrap_homo_tuple(retype):
    return retype.__args__[0]


def is_value_union(retype) -> bool:
    if isinstance(retype, tuple):
        if all(not callable(t) for t in retype):
            return True
    return False


def unwrap_value_union(retype):
    return type(retype[0])


def is_type(retype) -> bool:
    return getattr(retype, '_name', None) == 'Type'


def unwrap_type(retype):
    return retype.__args__[0]


def is_function_union(retype) -> bool:
    if is_optional(retype):
        return is_function_union(unwrap_optional(retype))
    if is_type(retype):
        if is_union(retype) or callable(retype):
            return True
    return False


def unwrap_function_union(retype):
    if is_union(retype.__args__[0]):
        return retype.__args__[0].__args__
    return retype.__args__


def get_annotations(func: Callable, only_with_default: bool = False):
    if inspect.isclass(func) or inspect.ismethod(func):
        remove_first = True
    else:
        remove_first = False

    annotations = typing.get_type_hints(func)
    spec = inspect.getfullargspec(func)
    args = spec.args
    defaults = spec.defaults or []

    if remove_first:
        args = args[1:]

    if only_with_default:
        return [(name, annotations.get(name, str), default, f'{name}')
                for name, default in zip(args[::-1], defaults[::-1])
                ][::-1]
    else:
        return [(name, annotations.get(name, str), default, f'{name}')
                for name, default in zip_longest(args[::-1], defaults[::-1], fillvalue=SUPPRESS)
                ][::-1]
