from argparse import SUPPRESS, ArgumentParser
from inspect import getfullargspec
from itertools import zip_longest
from typing import get_type_hints


def fetch_annotations(tp):
    arg_spec = getfullargspec(tp)
    type_hints = get_type_hints(tp)

    name_default = zip_longest(
        reversed(arg_spec.args),
        reversed(arg_spec.defaults or []),
        fillvalue=SUPPRESS,
    )
    for name, default in reversed(list(name_default)):
        yield name, default, type_hints[name]


def fetch_actions(argument_parser: ArgumentParser) -> str:
    msg = ', '.join([
        action.option_strings[-1]
        for action in argument_parser._actions
    ])
    return f"[{msg}]"
