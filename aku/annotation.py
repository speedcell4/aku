import inspect
from argparse import SUPPRESS
from itertools import zip_longest
from typing import Optional, TypeVar, Union


def add(x: int, y: int = None) -> int:
    return x + y


class Po(object):
    def __init__(self, x: int = 2, y: int = None):
        self.x = x
        self.y = y


class Rec(object):
    def __init__(self, x: int, y: int, w: int, h: int):
        self.x = x
        self.y = y
        self.w = w
        self.h = h


def is_list(ty):
    return getattr(ty, '_name', None) == 'List'


def is_tuple(ty):
    if getattr(ty, '_name', None) == 'Tuple':
        args = getattr(ty, '__args__')
        if len(args) == 2 and args[-1] is ...:
            return True
    return False


def is_type_var(ty):
    return isinstance(ty, TypeVar)


def is_callable(ty):
    return getattr(ty, '_name', None) == 'Callable'


def is_value_union(ty):
    if isinstance(ty, tuple):
        if not any(callable(t) for t in ty):
            return True
    return False


def is_type_union(ty):
    if getattr(ty, '__origin__', None) is Union:
        args = getattr(ty, '__args__')
        if type(None) in args:
            return False
        return True
    return False


def is_optional(ty):
    if getattr(ty, '__origin__', None) is Union:
        args = getattr(ty, '__args__')
        if type(None) in args:
            return True
    return False


def list_arg(ty):
    return getattr(ty, '__args__')[0]


def tuple_arg(ty):
    return getattr(ty, '__args__')[0]


def type_var_args(ty):
    return getattr(ty, '__constraints__')


def value_union_arg(ty):
    return type(ty[0])


def type_union_args(ty):
    return getattr(ty, '__args__')


def optional_args(ty):
    ret = Union[getattr(ty, '__args__')[:-1]]
    if is_type_union(ret):
        return type_union_args(ret)
    return ret


def annotation_iter(func):
    spec = inspect.getfullargspec(func)
    args = spec.args
    if inspect.ismethod(func) or inspect.isclass(func):
        args = spec.args[1:]

    names, annotations, defaults = [], [], []
    for name, default in zip_longest(reversed(args), reversed(spec.defaults or []), fillvalue=SUPPRESS):
        names.append(name)
        annotation = spec.annotations.get(name, str)
        annotations.append(Optional[annotation] if default is None else annotation)
        defaults.append(default)

    return zip(names[::-1], annotations[::-1], defaults[::-1])


_types = {}


def register_type(func):
    ty = inspect.getfullargspec(func).annotations.get('return')
    if ty not in _types:
        _types[ty] = func
    return func


@register_type
def str2none(option_string: str) -> type(None):
    option_string = option_string.lower()
    if option_string in ('nil', 'none', 'null'):
        return None
    raise ValueError(f'{option_string} is not a null value')


@register_type
def str2bool(option_string: str) -> bool:
    option_string = option_string.lower()
    if option_string in ('1', 'y', 'yes', 't', 'true'):
        return True
    if option_string in ('0', 'n', 'no', 'f', 'false'):
        return False
    raise ValueError(f'{option_string} is not a boolean value')


def combine_types(*fns):
    @register_type
    def type_fn(option_string: str) -> Union[fns]:
        for fn in fns:
            try:
                return fn(option_string)
            except ValueError:
                pass
        raise ValueError

    return type_fn


def get_type(ty):
    if is_list(ty):
        return list_type(ty)
    if is_tuple(ty):
        return tuple_type(ty)
    if is_value_union(ty):
        return value_union_type(ty)
    if is_type_union(ty):
        return type_union_type(ty)
    if is_type_var(ty):
        return type_var_type(ty)
    if is_optional(ty):
        return optional_type(ty)
    return primitive_type(ty)


def primitive_type(ty):
    return _types.get(ty, ty)


def list_type(ty):
    return get_type(list_arg(ty))


def tuple_type(ty):
    return get_type(tuple_arg(ty))


def value_union_type(ty):
    return get_type(value_union_arg(ty))


def type_union_type(ty):
    return str


def type_var_type(ty):
    return str


def optional_type(ty):
    return combine_types(
        get_type(optional_args(ty)),
        get_type(type(None))
    )


def get_metavar(ty):
    if is_list(ty):
        return list_metavar(ty)
    if is_tuple(ty):
        return type_union_metavar(ty)
    if is_value_union(ty):
        return value_union_metavar(ty)
    if is_type_union(ty):
        return type_union_metavar(ty)
    if is_type_var(ty):
        return type_var_metavar(ty)
    if is_optional(ty):
        return optional_metavar(ty)
    return primitive_metavar(ty)


def list_metavar(ty):
    return f'[{get_metavar(list_arg(ty))}]'


def tuple_metavar(ty):
    return f'({get_metavar(tuple_arg(ty))})'


def value_union_metavar(ty):
    return None


def type_union_metavar(ty):
    return None


def type_var_metavar(ty):
    return None


def optional_metavar(ty):
    return f'{get_metavar(optional_args(ty))}?'


def primitive_metavar(ty):
    return ty.__name__
