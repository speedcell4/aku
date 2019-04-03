import inspect
import typing
from argparse import SUPPRESS, ArgumentTypeError
from itertools import zip_longest
from typing import Callable, Optional, TypeVar, Union

NoneType = type(None)


def is_optional(retype) -> bool:
    if getattr(retype, '__origin__', None) is typing.Union:
        if NoneType in retype.__args__ and len(retype.__args__) == 2:
            return True  # Optional[T]
        return False
    return False


def unwrap_optional(retype):
    return Union[tuple(ty for ty in retype.__args__ if ty is not NoneType)]


def is_union(retype) -> bool:
    if getattr(retype, '__origin__', None) is typing.Union:
        if NoneType in retype.__args__ and len(retype.__args__) == 2:
            return False  # Optional[T]
        return True
    return False


def unwrap_union(retype):
    return retype.__args__


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
        T = type(retype[0])
        if any(not isinstance(item, T) for item in retype):
            raise ArgumentTypeError(f'value union :: {retype} requires all of its candidates be the same type')
    return False


def unwrap_value_union(retype):
    return type(retype[0])


def is_type(retype) -> bool:
    return getattr(retype, '_name', None) == 'Type'


def unwrap_type(retype):
    return retype.__args__[0]


def is_function_union(retype) -> bool:
    if is_type(retype):
        retype = unwrap_type(retype)
        if is_union(retype) or callable(retype):
            return True
    return False


def unwrap_function_union(retype):
    if is_union(retype.__args__[0]):
        return retype.__args__[0].__args__
    return retype.__args__


def is_type_var(retype):
    return isinstance(retype, TypeVar)


def unwrap_type_var(retype):
    return retype.__constraints__


def get_annotations(func: Callable):
    if inspect.isclass(func) or inspect.ismethod(func):
        remove_first = True
    else:
        remove_first = False

    spec = inspect.getfullargspec(func)
    args, defaults, annotations = spec.args, spec.defaults, spec.annotations
    if defaults is None:
        defaults = []

    if remove_first:
        args = args[1:]

    ret = []
    for name, default in zip_longest(args[::-1], defaults[::-1], fillvalue=SUPPRESS):
        annotation = annotations.get(name, str)
        if default is None:
            annotation = Optional[annotation]
        ret.append((name, annotation, default, f'{name}'))
    return ret[::-1]


def render_type(retype) -> Optional[str]:
    if is_optional(retype):
        args = render_type(unwrap_optional(retype))
        return f'{args}?'

    if is_union(retype):
        args = ','.join(render_type(a) for a in unwrap_union(retype))
        return f'{{{args}}}'
    if is_list(retype):
        args = render_type(unwrap_list(retype))
        return f'[{args}]'
    if is_homo_tuple(retype):
        args = render_type(unwrap_homo_tuple(retype))
        return f'({args})'
    if is_value_union(retype):
        return None

    return f'{retype.__name__}'.capitalize()
