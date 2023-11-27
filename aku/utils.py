import inspect
import re
from argparse import ArgumentParser
from argparse import SUPPRESS
from inspect import Parameter
from typing import FrozenSet
from typing import Pattern
from typing import Set
from typing import Tuple
from typing import get_type_hints

AKU = '@aku'
AKU_FN = '@fn'
AKU_DELAY = '@delay'
AKU_VISITED = '@visited'


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


def register_homo_tuple_type(tp: type, argument_parser: ArgumentParser,
                             pattern: Pattern = re.compile(r',\s*')) -> None:
    def fn(string: str) -> Tuple[tp, ...]:
        nonlocal tp

        tp = argument_parser._registry_get('type', tp, tp)
        return tuple(tp(arg) for arg in re.split(pattern, string.strip()))

    return register_type(fn, argument_parser)


def register_hetero_tuple_type(tps: Tuple[type, ...], argument_parser: ArgumentParser,
                               pattern: Pattern = re.compile(r',\s*')) -> None:
    def fn(string: str) -> Tuple[tps]:
        nonlocal tps

        tps = [argument_parser._registry_get('type', tp, tp) for tp in tps]
        args = re.split(pattern, string.strip())

        if len(tps) != len(args):
            raise ValueError(f'the number of arguments does not match, {len(tps)} != {len(args)}')
        return tuple(tp(arg) for tp, arg in zip(tps, args))

    return register_type(fn, argument_parser)


def register_set_type(tp: type, argument_parser: ArgumentParser,
                      pattern: Pattern = re.compile(r',\s*')) -> None:
    def fn(string: str) -> Set[tp]:
        nonlocal tp

        tp = argument_parser._registry_get('type', tp, tp)
        return set(tp(arg) for arg in re.split(pattern, string.strip()))

    return register_type(fn, argument_parser)


def register_frozenset_type(tp: type, argument_parser: ArgumentParser,
                            pattern: Pattern = re.compile(r',\s*')) -> None:
    def fn(string: str) -> FrozenSet[tp]:
        nonlocal tp

        tp = argument_parser._registry_get('type', tp, tp)
        return frozenset(tp(arg) for arg in re.split(pattern, string.strip()))

    return register_type(fn, argument_parser)


def iter_annotations(tp, positional_only: bool = False, positional_or_keyword: bool = True, keyword_only: bool = False):
    kinds = set()
    if positional_only:
        kinds.add(Parameter.POSITIONAL_ONLY)
    if positional_or_keyword:
        kinds.add(Parameter.POSITIONAL_OR_KEYWORD)
    if keyword_only:
        kinds.add(Parameter.KEYWORD_ONLY)

    for name, param in inspect.signature(tp).parameters.items():  # type: (str, Parameter)
        if param.kind in kinds:
            if param.annotation == inspect.Signature.empty:
                raise RuntimeError(f'parameter {name} requires an type annotation ({tp})')

            if param.default == Parameter.empty:
                yield name, param.annotation, SUPPRESS
            else:
                yield name, param.annotation, param.default


def get_action_group(argument_parser: ArgumentParser, title: str):
    argument_parser = getattr(argument_parser, 'container', argument_parser)

    for action_group in argument_parser._action_groups:
        if action_group.title == title:
            return action_group

    action_group = argument_parser.add_argument_group(title=title)
    action_group.container = argument_parser
    return argument_parser, action_group


def get_name(tp: type) -> str:
    try:
        return tp.__qualname__
    except AttributeError:
        return tp.__class__.__qualname__


def get_dest(domain: Tuple[str, ...], name: str) -> str:
    return '.'.join(domain + (name,)).lower()


def get_option(domain: Tuple[str, ...], name: str) -> str:
    prefix = tuple(d[:-1] for d in domain if d.endswith('_'))
    return '-'.join(prefix + (name[:-1] if name.endswith('_') else name,)).lower().replace('_', '-')
