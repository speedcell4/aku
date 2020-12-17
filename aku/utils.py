import inspect
import re
from argparse import ArgumentParser, SUPPRESS
from typing import get_type_hints, Pattern, Tuple

AKU = '@aku'
AKU_FN = '@fn'
AKU_ROOT = '@root'


def tp_bool(arg_strings: str) -> bool:
    arg_strings = arg_strings.lower().strip()
    if arg_strings in ('t', 'true', 'y', 'yes', '1'):
        return True
    if arg_strings in ('f', 'false', 'n', 'no', '0'):
        return False
    raise ValueError


def register_type(fn, argument_parser: ArgumentParser):
    tp = get_type_hints(fn)['return']
    registry = argument_parser._registries['type']
    if tp not in registry:
        registry.setdefault(tp, fn)
    return fn


def register_homo_tuple(tp: type, argument_parser: ArgumentParser,
                        pattern: Pattern = re.compile(r',\s*')) -> None:
    def fn(arg_strings: str) -> Tuple[tp, ...]:
        nonlocal tp

        tp = argument_parser._registry_get('type', tp, tp)
        return tuple(tp(arg) for arg in re.split(pattern, arg_strings.strip()))

    return register_type(fn, argument_parser)


def register_hetero_tuple(tps: Tuple[type, ...], argument_parser: ArgumentParser,
                          pattern: Pattern = re.compile(r',\s*')) -> None:
    def fn(arg_strings: str) -> Tuple[tps]:
        nonlocal tps

        tps = [argument_parser._registry_get('type', tp, tp) for tp in tps]
        args = re.split(pattern, arg_strings.strip())

        if len(tps) != len(args):
            raise ValueError(f'the number of arguments does not match, {len(tps)} != {len(args)}')
        return tuple(tp(arg) for tp, arg in zip(tps, args))

    return register_type(fn, argument_parser)


def _init_argument_parser(argument_parser: ArgumentParser):
    register_type(tp_bool, argument_parser)


def tp_iter(fn):
    is_method = inspect.ismethod(fn)
    if inspect.isclass(fn):
        fn = fn.__init__
        is_method = True

    tps = get_type_hints(fn)
    spec = inspect.getfullargspec(fn)
    args = spec.args or []
    defaults = spec.defaults or []
    defaults = {a: d for a, d in zip(args[::-1], defaults[::-1])}

    for index, arg in enumerate(args[1:] if is_method else args):
        yield arg, tps[arg], defaults.get(arg, SUPPRESS)


def fetch_name(fn) -> str:
    if inspect.isfunction(fn):  # function, static method
        return fn.__name__
    if inspect.isclass(fn):  # class
        return fn.__name__.lower()
    if inspect.ismethod(fn):  # class method
        __class__ = fn.__self__
        if not inspect.isclass(__class__):
            __class__ = __class__.__class__

        return f'{__class__.__name__.lower()}.{fn.__name__}'
    if callable(fn):  # __call__
        return f'{fn.__class__.__name__.lower()}'
    raise NotImplementedError


def join_names(prefixes: Tuple[str, ...], name: str) -> str:
    if name.endswith('_'):
        name = name[:-1]
    return '-'.join(prefixes + (name,)).lower()


def join_dests(domain: Tuple[str, ...], name: str) -> str:
    return '.'.join(domain + (name,)).lower()


def get_action_group(argument_parser: ArgumentParser, title: str):
    argument_parser = getattr(argument_parser, 'container', argument_parser)

    for action_group in argument_parser._action_groups:
        if action_group.title == title:
            return action_group

    action_group = argument_parser.add_argument_group(title=title)
    action_group.container = argument_parser
    return argument_parser, action_group
