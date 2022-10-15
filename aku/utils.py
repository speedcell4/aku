import inspect
import re
from argparse import ArgumentParser, SUPPRESS
from inspect import Parameter
from typing import get_type_hints, Pattern, Tuple, Set, FrozenSet, Callable

AKU = '@aku'
AKU_FN = '@fn'
AKU_DELAY = '@delay'
AKU_VISITED = '@visited'
AKU_NAME = '__aku_name__'


def bool_type(string: str) -> bool:
    string = string.lower().strip()
    if string in ('t', 'true', 'y', 'yes', '1'):
        return True
    if string in ('f', 'false', 'n', 'no', '0'):
        return False
    raise ValueError


def register_type(fn, argument_parser: ArgumentParser):
    tp = get_type_hints(fn)['return']
    registry = argument_parser._registries['type']
    if tp not in registry:
        registry.setdefault(tp, fn)
    return fn


def register_homo_tuple_type(tp: type, argument_parser: ArgumentParser, pattern: Pattern = re.compile(r',\s*')) -> None:
    def fn(string: str) -> Tuple[tp, ...]:
        nonlocal tp

        tp = argument_parser._registry_get('type', tp, tp)
        return tuple(tp(arg) for arg in re.split(pattern, string.strip()))

    return register_type(fn, argument_parser)


def register_hetero_tuple(tps: Tuple[type, ...], argument_parser: ArgumentParser,
                          pattern: Pattern = re.compile(r',\s*')) -> None:
    def fn(string: str) -> Tuple[tps]:
        nonlocal tps

        tps = [argument_parser._registry_get('type', tp, tp) for tp in tps]
        args = re.split(pattern, string.strip())

        if len(tps) != len(args):
            raise ValueError(f'the number of arguments does not match, {len(tps)} != {len(args)}')
        return tuple(tp(arg) for tp, arg in zip(tps, args))

    return register_type(fn, argument_parser)


def register_set_type(tp: type, argument_parser: ArgumentParser, pattern: Pattern = re.compile(r',\s*')) -> None:
    def fn(string: str) -> Set[tp]:
        nonlocal tp

        tp = argument_parser._registry_get('type', tp, tp)
        return set(tp(arg) for arg in re.split(pattern, string.strip()))

    return register_type(fn, argument_parser)


def register_frozenset_type(tp: type, argument_parser: ArgumentParser, pattern: Pattern = re.compile(r',\s*')) -> None:
    def fn(string: str) -> FrozenSet[tp]:
        nonlocal tp

        tp = argument_parser._registry_get('type', tp, tp)
        return frozenset(tp(arg) for arg in re.split(pattern, string.strip()))

    return register_type(fn, argument_parser)


def iter_annotations(tp):
    for name, param in inspect.signature(tp).parameters.items():  # type: (str, Parameter)
        if param.kind in (Parameter.POSITIONAL_OR_KEYWORD, Parameter.POSITIONAL_ONLY):
            if param.default == Parameter.empty:
                yield name, param.annotation, SUPPRESS
            else:
                yield name, param.annotation, param.default


def rename(name: str):
    def wrap(tp: Callable) -> Callable:
        assert not hasattr(tp, AKU_NAME), f'{tp} is already renamed to {getattr(tp, AKU_NAME)}'
        setattr(tp, AKU_NAME, name)
        return tp

    return wrap


def get_name(tp: Callable) -> str:
    try:
        return getattr(tp, AKU_NAME, tp.__qualname__)
    except AttributeError:
        return tp.__class__.__qualname__


def join_name(prefix: Tuple[str, ...], name: str) -> str:
    if name.endswith('_'):
        name = name[:-1]
    return '-'.join(prefix + (name,)).lower().replace('_', '-')


def join_dest(domain: Tuple[str, ...], name: str) -> str:
    return '.'.join(domain + (name,)).lower()


def get_action_group(argument_parser: ArgumentParser, title: str):
    argument_parser = getattr(argument_parser, 'container', argument_parser)

    for action_group in argument_parser._action_groups:
        if action_group.title == title:
            return action_group

    action_group = argument_parser.add_argument_group(title=title)
    action_group.container = argument_parser
    return argument_parser, action_group
