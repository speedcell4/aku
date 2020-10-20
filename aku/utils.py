import inspect
import re
from argparse import ArgumentParser, SUPPRESS
from typing import get_type_hints, Pattern, Tuple

NEW_ACTIONS = '_new_actions'


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
        return tuple(tp(arg) for tp, arg in zip(tps, re.split(pattern, arg_strings.strip())))

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


def join_names(prefixes: Tuple[str, ...], name: str) -> str:
    if name.endswith('_'):
        name = name[:-1]
    return '-'.join(prefixes + (name,)).lower()


def join_dests(domain: Tuple[str, ...], name: str) -> str:
    return '.'.join(domain + (name,)).lower()
