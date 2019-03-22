from argparse import Action, ArgumentParser, ArgumentTypeError, Namespace
from typing import List, Tuple

from aku.metavars import render_type
from aku.parsers import get_parsing_fn
from aku.utils import get_annotations, is_function_union, is_homo_tuple, is_list, is_value_union

EXECUTED = '__AKU_EXECUTED'


def append_prefix(prefix1, prefix2):
    if prefix1 is None:
        return prefix2
    if prefix2 is None:
        return prefix1
    return f'{prefix1}_{prefix2}'


def get_dest_name(name, prefix):
    if prefix is not None:
        return f'{prefix}_{name}'
    return f'{name}'


_argument_fn = []


def register_argument_fn(func):
    _argument_fn.append(func)
    return func


@register_argument_fn
def add_primitive(parser: ArgumentParser, annotation: Tuple, prefix: str, **kwargs):
    name, annotation, default, desc = annotation
    dest_name = get_dest_name(name, prefix)

    parser.add_argument(
        f'--{dest_name}', default=default, help=desc,
        type=get_parsing_fn(annotation), metavar=render_type(annotation),
    )

    return dest_name


class ListAppendAction(Action):
    def __call__(self, parser: ArgumentParser, namespace: Namespace, values, option_string) -> None:
        if not getattr(self, EXECUTED, False):
            setattr(self, EXECUTED, True)
            setattr(namespace, self.dest, [])
        setattr(namespace, self.dest, [*getattr(namespace, self.dest), values])


@register_argument_fn
def add_list(parser: ArgumentParser, annotation: Tuple, prefix: str, **kwargs):
    name, annotation, default, desc = annotation
    dest_name = get_dest_name(name, prefix)
    if not is_list(annotation):
        raise TypeError

    retype = annotation.__args__[0]
    parser.add_argument(
        f'--{dest_name}', default=default, help=desc,
        type=get_parsing_fn(retype), metavar=render_type(annotation),
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
def add_homo_tuple(parser: ArgumentParser, annotation: Tuple, prefix: str, **kwargs):
    name, annotation, default, desc = annotation
    dest_name = get_dest_name(name, prefix)
    if not is_homo_tuple(annotation):
        raise TypeError

    retype = annotation.__args__[0]
    parser.add_argument(
        f'--{dest_name}', default=default, help=desc,
        type=get_parsing_fn(retype), metavar=render_type(annotation),
        action=HomoTupleAppendAction,
    )

    return dest_name


@register_argument_fn
def add_value_union(parser: ArgumentParser, annotation: Tuple, prefix: str, **kwargs):
    name, annotation, default, desc = annotation
    dest_name = get_dest_name(name, prefix)
    if not is_value_union(annotation):
        raise TypeError

    retype = type(annotation[0])
    parser.add_argument(
        f'--{dest_name}', default=default, help=desc,
        type=get_parsing_fn(retype), choices=annotation,
    )

    return dest_name


@register_argument_fn
def add_function_union(parser: ArgumentParser, annotation: Tuple, prefix: str, **kwargs):
    name, annotation, default, desc = annotation
    dest_name = get_dest_name(f'{name}', prefix)
    choose_dest_name = get_dest_name(f'{name}_choose', prefix)
    if not is_function_union(annotation):
        raise TypeError

    function_map = {
        f.__name__: f
        for f in annotation
    }

    class ChooseFunctionAction(Action):
        def __init__(self, *args, prefix, group, **kwargs):
            super(ChooseFunctionAction, self).__init__(*args, **kwargs)
            self.prefix = prefix
            self.group = group

        def __call__(self, parser: ArgumentParser, namespace: Namespace, values, option_string) -> None:
            if not getattr(self, EXECUTED, False):
                setattr(self, EXECUTED, True)
                setattr(namespace, self.dest, values)
                names = add_argument(
                    func=function_map[values], parser=self.group,
                    prefix=append_prefix(prefix, self.prefix),
                    **kwargs,
                )
                parser.set_defaults(**{
                    dest_name: function_map[values],
                })
                kwargs.get('slots').append((dest_name, names))
                parser.parse_known_args()

    group = parser.add_argument_group(name)
    group.add_argument(
        f'--{dest_name}', dest=choose_dest_name, default=default, help=desc,
        choices=function_map.keys(), action=ChooseFunctionAction,
        type=get_parsing_fn(str), prefix=name, group=group,
    )

    return dest_name


def add_argument(func, parser: ArgumentParser, prefix: str = None, **kwargs) -> List:
    names = []
    for annotation in get_annotations(func):
        ok = False
        for argument_fn in _argument_fn[::-1]:
            try:
                name = argument_fn(parser=parser, annotation=annotation, prefix=prefix, **kwargs)
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
