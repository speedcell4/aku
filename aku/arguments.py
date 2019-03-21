from argparse import Action, ArgumentParser, Namespace
from typing import Tuple

from aku.metavars import render_type
from aku.parsers import get_parsing_fn
from aku.utils import get_annotations, is_homo_tuple, is_list, is_value_union

EXECUTED = '__AKU_EXECUTED'


def append_prefix(prefix1, prefix2):
    if prefix1 is None:
        return prefix2
    if prefix2 is None:
        return prefix1
    return f'{prefix1}-{prefix2}'


def get_option_name(name, prefix):
    if prefix is not None:
        return f'--{prefix}-{name}'
    return f'--{name}'


class Argument(object):
    _none_primitive_arguments = []

    def __init__(self, name, parsing_fn, metavar, default, desc, action='store', choices=None):
        super(Argument, self).__init__()
        self.name = name
        self.parsing_fn = parsing_fn
        self.metavar = metavar
        self.default = default
        self.desc = desc
        self.action = action
        self.choices = choices

    def __init_subclass__(cls, **kwargs):
        cls._none_primitive_arguments.append(cls)

    def __call__(self, parser: ArgumentParser, prefix, *args, **kwargs):
        return parser.add_argument(
            get_option_name(self.name, prefix), action=self.action,
            default=self.default, type=self.parsing_fn,
            help=self.desc, metavar=self.metavar, choices=self.choices,
        )

    def __class_getitem__(cls, annotation: Tuple) -> 'Argument':
        for argument in cls._none_primitive_arguments:
            try:
                return argument[annotation]
            except ValueError:
                pass

        name, annotation, default, desc = annotation
        return cls(
            name=name, default=default, desc=desc,
            parsing_fn=get_parsing_fn(annotation),
            metavar=render_type(annotation),
        )


class ListAppendAction(Action):
    def __call__(self, parser: ArgumentParser, namespace: Namespace, values, option_string) -> None:
        if not getattr(self, EXECUTED, False):
            setattr(self, EXECUTED, True)
            setattr(namespace, self.dest, [])
        setattr(namespace, self.dest, [*getattr(namespace, self.dest), values])


class ListArgument(Argument):
    def __init__(self, *args, **kwargs):
        super(ListArgument, self).__init__(*args, **kwargs)
        self.action = ListAppendAction

    def __class_getitem__(cls, annotation: Tuple) -> 'Argument':
        name, annotation, default, desc = annotation
        if not is_list(annotation):
            raise ValueError

        retype = annotation.__args__[0]
        return cls(
            name=name, default=default, desc=desc,
            parsing_fn=get_parsing_fn(retype),
            metavar=render_type(annotation),
        )


class TupleAppendAction(Action):
    def __call__(self, parser: ArgumentParser, namespace: Namespace, values, option_string) -> None:
        if not getattr(self, EXECUTED, False):
            setattr(self, EXECUTED, True)
            setattr(namespace, self.dest, ())
        setattr(namespace, self.dest, (*getattr(namespace, self.dest), values))


class HomoTupleArgument(Argument):
    def __init__(self, *args, **kwargs):
        super(HomoTupleArgument, self).__init__(*args, **kwargs)
        self.action = TupleAppendAction

    def __class_getitem__(cls, annotation: Tuple) -> 'Argument':
        name, annotation, default, desc = annotation
        if not is_homo_tuple(annotation):
            raise ValueError

        retype = annotation.__args__[0]
        return cls(
            name=name, default=default, desc=desc,
            parsing_fn=get_parsing_fn(retype),
            metavar=render_type(annotation),
        )


class ValueUnionArgument(Argument):
    def __class_getitem__(cls, annotation: Tuple) -> 'Argument':
        name, annotation, default, desc = annotation
        if not is_value_union(annotation):
            raise ValueError

        retype = type(annotation[0])
        return cls(
            name=name, default=default, desc=desc,
            parsing_fn=get_parsing_fn(retype),
            metavar=render_type(annotation), choices=annotation,
        )


def expand_function(func, parser: ArgumentParser, prefix: str = None, *args, **kwargs):
    for annotation in get_annotations(func):
        Argument[annotation](parser=parser, prefix=prefix, *args, **kwargs)
