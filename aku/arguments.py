from argparse import ArgumentParser
from typing import Tuple

from aku.metavars import render_type
from aku.parsers import get_parsing_fn
from aku.utils import get_annotations


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

    def __init__(self, name, parsing_fn, metavar, default, desc):
        super(Argument, self).__init__()
        self.name = name
        self.parsing_fn = parsing_fn
        self.metavar = metavar
        self.default = default
        self.desc = desc

    def __init_subclass__(cls, **kwargs):
        cls._none_primitive_arguments.append(cls)

    def __call__(self, parser: ArgumentParser, prefix, *args, **kwargs):
        return parser.add_argument(
            get_option_name(self.name, prefix), action='store',
            default=self.default, type=self.parsing_fn,
            required=True, help=self.desc, metavar=self.metavar,
        )

    def __class_getitem__(cls, annotation: Tuple) -> 'Argument':
        for argument in cls._none_primitive_arguments:
            try:
                return argument(*annotation)
            except ValueError:
                pass

        name, annotation, default, desc = annotation
        print(f'annotation => {annotation}')
        return cls(
            name=name, default=default, desc=desc,
            parsing_fn=get_parsing_fn(annotation),
            metavar=render_type(annotation),
        )


def expand_function(func, parser: ArgumentParser, prefix: str, *args, **kwargs):
    for annotation in get_annotations(func):
        argument = Argument[annotation]
        argument(parser=parser, prefix=prefix, *args, **kwargs)
