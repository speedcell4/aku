import inspect
from argparse import Action, ArgumentParser, ArgumentTypeError, Namespace
from typing import List, Optional, Tuple

from aku.parsers import get_parsing_fn
from aku.utils import get_annotations, is_homo_tuple, is_list, is_type_union, is_type_var, is_value_union, metavar, \
    unwrap_homo_tuple, unwrap_list, unwrap_type_union, unwrap_type_var, unwrap_value_union

EXECUTED = '_AKU_EXECUTED'


def append_prefix(prefix1: Optional[str], prefix2: Optional[str]) -> Optional[str]:
    if prefix1 is None:
        return prefix2
    if prefix2 is None:
        return prefix1
    return f'{prefix1}_{prefix2}'


def get_dest_name(name: str, prefix: Optional[str]) -> str:
    if prefix is not None:
        return f'{prefix}_{name}'
    return f'{name}'


_argument_fn = []


def register_argument_fn(func):
    _argument_fn.append(func)
    return func


@register_argument_fn
def add_primitive(parser: ArgumentParser, group: ArgumentParser,
                  annotation: Tuple, prefix: str, **kwargs):
    name, annotation, default, desc = annotation
    dest_name = get_dest_name(name, prefix)

    group.add_argument(
        f'--{dest_name}', default=default, help=desc,
        type=get_parsing_fn(annotation), metavar=metavar(annotation),
    )

    return dest_name


class ListAppendAction(Action):
    def __call__(self, parser: ArgumentParser, namespace: Namespace, values, option_string) -> None:
        if not getattr(self, EXECUTED, False):
            setattr(self, EXECUTED, True)
            setattr(namespace, self.dest, [])
        setattr(namespace, self.dest, [*getattr(namespace, self.dest), values])


@register_argument_fn
def add_list(parser: ArgumentParser, group: ArgumentParser,
             annotation: Tuple, prefix: str, **kwargs):
    name, annotation, default, desc = annotation
    dest_name = get_dest_name(name, prefix)
    if not is_list(annotation):
        raise TypeError

    retype = unwrap_list(annotation)
    group.add_argument(
        f'--{dest_name}', default=default, help=desc,
        type=get_parsing_fn(retype), metavar=metavar(annotation),
        action=ListAppendAction,
    )

    return dest_name


class HomoTupleAppendAction(Action):
    def __call__(self, parser: ArgumentParser, namespace: Namespace, values, option_string) -> None:
        if not getattr(self, EXECUTED, False):
            setattr(self, EXECUTED, True)
            setattr(namespace, self.dest, ())
        setattr(namespace, self.dest, (*getattr(namespace, self.dest), values))


@register_argument_fn
def add_homo_tuple(parser: ArgumentParser, group: ArgumentParser,
                   annotation: Tuple, prefix: str, **kwargs):
    name, annotation, default, desc = annotation
    dest_name = get_dest_name(name, prefix)
    if not is_homo_tuple(annotation):
        raise TypeError

    retype = unwrap_homo_tuple(annotation)
    group.add_argument(
        f'--{dest_name}', default=default, help=desc,
        type=get_parsing_fn(retype), metavar=metavar(annotation),
        action=HomoTupleAppendAction,
    )

    return dest_name


@register_argument_fn
def add_value_union(parser: ArgumentParser, group: ArgumentParser,
                    annotation: Tuple, prefix: str, **kwargs):
    name, annotation, default, desc = annotation
    dest_name = get_dest_name(name, prefix)
    if not is_value_union(annotation):
        raise TypeError

    retype = unwrap_value_union(annotation)
    group.add_argument(
        f'--{dest_name}', default=default, help=desc,
        type=get_parsing_fn(retype), choices=annotation,
    )

    return dest_name


@register_argument_fn
def add_type_union(parser: ArgumentParser, group: ArgumentParser,
                   annotation: Tuple, prefix: str, **kwargs):
    name, annotation, default, desc = annotation
    option_name = get_dest_name(f'{name}', prefix)
    dest_name = get_dest_name(f'{name}_choose', prefix)

    if not is_type_union(annotation):
        raise TypeError

    group = parser.add_argument_group(name)
    function_map = {
        f.__self__.__name__ if inspect.ismethod(f) else f.__name__: f
        for f in unwrap_type_union(annotation)
    }

    class ChooseFunctionAction(Action):
        def __init__(self, *args, **kwargs):
            super(ChooseFunctionAction, self).__init__(*args, **kwargs)

        def __call__(self, parser: ArgumentParser, namespace: Namespace, values, option_string) -> None:
            if not getattr(self, EXECUTED, False):
                setattr(self, EXECUTED, True)
                setattr(namespace, self.dest, values)
                names = add_function(
                    parser=parser, group=group,
                    func=function_map[values],
                    prefix=prefix, **kwargs,
                )
                parser.set_defaults(**{
                    option_name: function_map[values],
                })
                if option_name != names:
                    kwargs.get('slots').append((option_name, names))
                parser.parse_known_args()

    group.add_argument(
        f'--{option_name}', dest=dest_name, default=default, help=desc,
        type=get_parsing_fn(str), choices=list(function_map.keys()), action=ChooseFunctionAction,
    )

    return option_name


@register_argument_fn
def add_type_var(parser: ArgumentParser, group: ArgumentParser,
                 annotation: Tuple, prefix: str, **kwargs):
    name, annotation, default, desc = annotation
    option_name = get_dest_name(f'{name}', prefix)
    dest_name = get_dest_name(f'{name}_choose', prefix)

    if not is_type_var(annotation):
        raise TypeError

    group = parser.add_argument_group(name)
    function_map = {
        f.__self__.__name__ if inspect.ismethod(f) else f.__name__: f
        for f in unwrap_type_var(annotation)
    }

    class ChooseFunctionAction(Action):
        def __init__(self, *args, **kwargs):
            super(ChooseFunctionAction, self).__init__(*args, **kwargs)

        def __call__(self, parser: ArgumentParser, namespace: Namespace, values, option_string) -> None:
            if not getattr(self, EXECUTED, False):
                setattr(self, EXECUTED, True)
                setattr(namespace, self.dest, values)
                names = add_function(
                    parser=parser, group=group,
                    func=function_map[values],
                    prefix=append_prefix(prefix, annotation.__name__), **kwargs,
                )
                parser.set_defaults(**{
                    option_name: function_map[values],
                })
                if option_name != names:
                    kwargs.get('slots').append((option_name, names))
                parser.parse_known_args()

    group.add_argument(
        f'--{option_name}', dest=dest_name, default=default, help=desc,
        type=get_parsing_fn(str), choices=list(function_map.keys()), action=ChooseFunctionAction,
    )

    return option_name


def add_function(parser: ArgumentParser, group: ArgumentParser,
                 func, prefix: str = None, **kwargs) -> List:
    names = []
    for annotation in get_annotations(func):
        ok = False
        for argument_fn in _argument_fn[::-1]:
            try:
                name = argument_fn(
                    parser=parser, arguments=group,
                    annotation=annotation, prefix=prefix, **kwargs,
                )
                names.append((annotation[0], name))
                ok = True
                break
            except TypeError:
                pass
        if not ok:
            raise ArgumentTypeError(
                f'"{annotation[0]} : {annotation[1]}" does not fail into any annotation category.'
            )
    return names
