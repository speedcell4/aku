import functools
import inspect
import re
from argparse import ArgumentParser, Action, Namespace, SUPPRESS, ArgumentDefaultsHelpFormatter
from re import Pattern
from typing import Union, Tuple, Literal, Any, Type
from typing import get_origin, get_args, get_type_hints

from aku.utils import get_max_help_position

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


class AkuTp(object):
    def __init__(self, tp, choices):
        super(AkuTp, self).__init__()
        self.tp = tp
        self.choices = choices

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.tp.__name__}, {self.choices})'

    registry = []

    def __init_subclass__(cls, **kwargs):
        cls.registry.append(cls)

    def __class_getitem__(cls, tp):
        origin = get_origin(tp)
        args = get_args(tp)
        for aku_ty in cls.registry:
            try:
                return aku_ty[tp, origin, args]
            except TypeError:
                pass
        raise TypeError(f'unsupported annotation {tp}')

    def add_argument(self, argument_parser: ArgumentParser, name: str, default: Any,
                     prefixes: Tuple[str, ...], domain: Tuple[str, ...]) -> None:
        raise NotImplementedError


class StoreAction(Action):
    def __call__(self, parser: ArgumentParser, namespace: Namespace, values, option_string=None):
        setattr(namespace, self.dest, values)
        self.required = False


class AkuPrimitive(AkuTp):
    def __class_getitem__(cls, tp):
        tp, origin, args = tp
        if origin is None:
            return AkuPrimitive(tp, None)
        raise TypeError

    def add_argument(self, argument_parser: ArgumentParser, name: str, default: Any,
                     prefixes: Tuple[str, ...], domain: Tuple[str, ...]) -> None:
        prefixes_name = join_names(prefixes, name)
        argument_parser.add_argument(
            f'--{prefixes_name}', dest=join_dests(domain, name), help=prefixes_name,
            type=self.tp, choices=self.choices, required=default == SUPPRESS,
            action=StoreAction, default=default, metavar=self.tp.__name__.lower(),
        )


class AppendListAction(Action):
    def __call__(self, parser: ArgumentParser, namespace: Namespace, values, option_string=None):
        if not getattr(self, '_aku_visited', False):
            setattr(self, '_aku_visited', True)
            setattr(namespace, self.dest, [])
        getattr(namespace, self.dest).append(values)
        self.required = False


class AkuList(AkuTp):
    def __class_getitem__(cls, tp):
        tp, origin, args = tp
        if origin is list:
            return AkuList(args[0], None)
        raise TypeError

    def add_argument(self, argument_parser: ArgumentParser, name: str, default: Any,
                     prefixes: Tuple[str, ...], domain: Tuple[str, ...]) -> None:
        prefixes_name = join_names(prefixes, name)
        argument_parser.add_argument(
            f'--{prefixes_name}', dest=join_dests(domain, name), help=prefixes_name,
            type=self.tp, choices=self.choices, required=default == SUPPRESS,
            action=AppendListAction, default=default, metavar=f'[{self.tp.__name__.lower()}]',
        )


class AkuHomoTuple(AkuTp):
    def __class_getitem__(cls, tp):
        tp, origin, args = tp
        if origin is tuple:
            if len(args) == 2 and args[1] is ...:
                return AkuHomoTuple(args[0], None)
            else:
                return AkuHeteroTuple(args, None)
        raise TypeError

    def add_argument(self, argument_parser: ArgumentParser, name: str, default: Any,
                     prefixes: Tuple[str, ...], domain: Tuple[str, ...]) -> None:
        prefixes_name = join_names(prefixes, name)
        argument_parser.add_argument(
            f'--{prefixes_name}', dest=join_dests(domain, name), help=prefixes_name,
            type=register_homo_tuple(self.tp, argument_parser), choices=self.choices, required=default == SUPPRESS,
            action=StoreAction, default=default, metavar=f'({self.tp.__name__.lower()}, ...)',
        )


class AkuHeteroTuple(AkuTp):
    def __class_getitem__(cls, tp):
        tp, origin, args = tp
        if origin is tuple:
            if len(args) == 2 and args[1] is ...:
                return AkuHomoTuple(args[0], None)
            else:
                return AkuHeteroTuple(args, None)
        raise TypeError

    def add_argument(self, argument_parser: ArgumentParser, name: str, default: Any,
                     prefixes: Tuple[str, ...], domain: Tuple[str, ...]) -> None:
        metavars = ', '.join(t.__name__.lower() for t in self.tp)
        prefixes_name = join_names(prefixes, name)
        argument_parser.add_argument(
            f'--{prefixes_name}', dest=join_dests(domain, name), help=prefixes_name,
            type=register_hetero_tuple(self.tp, argument_parser), choices=self.choices, required=default == SUPPRESS,
            action=StoreAction, default=default, metavar=f'({metavars})',
        )


class AkuLiteral(AkuTp):
    def __class_getitem__(cls, tp):
        tp, origin, args = tp
        if origin is Literal:
            if len(args) > 0:
                tp = type(args[0])
                for arg in args:
                    assert get_origin(arg) is None, f'{arg} is not a primitive type'
                    assert isinstance(arg, tp), f'{type(arg)} is not {tp}'
                return AkuLiteral(tp, args)
        raise TypeError

    def add_argument(self, argument_parser: ArgumentParser, name: str, default: Any,
                     prefixes: Tuple[str, ...], domain: Tuple[str, ...]) -> None:
        prefixes_name = join_names(prefixes, name)
        argument_parser.add_argument(
            f'--{prefixes_name}', dest=join_dests(domain, name), help=prefixes_name,
            type=self.tp, choices=self.choices, required=default == SUPPRESS,
            action=StoreAction, default=default, metavar=f'{self.tp.__name__.lower()}{set(self.choices)}',
        )


class AkuFn(AkuTp):
    def __class_getitem__(cls, tp):
        tp, origin, args = tp
        if origin is type:
            if len(args) == 1:
                if get_origin(args[0]) == Union:
                    return AkuUnion(str, get_args(args[0]))
                else:
                    return AkuFn(args[0], None)
        raise TypeError

    def add_argument(self, argument_parser: ArgumentParser, name: str, default: Any,
                     prefixes: Tuple[str, ...], domain: Tuple[str, ...]) -> None:

        if name is not None:
            domain = domain + (name,)
            if name.endswith('_'):
                prefixes = prefixes + (name[:-1],)

        for arg, tp, df in tp_iter(self.tp):
            tp = AkuTp[tp]
            tp.add_argument(
                argument_parser=argument_parser, name=arg,
                prefixes=prefixes, domain=domain, default=df,
            )


class AkuUnion(AkuTp):
    def __class_getitem__(cls, tp):
        tp, origin, args = tp
        if origin is type:
            if len(args) == 1:
                if get_origin(args[0]) == Union:
                    return AkuUnion(str, get_args(args[0]))
                else:
                    return AkuFn(args[0], None)
        elif origin is Union:
            args = [
                get_args(arg)[0]
                for arg in get_args(tp)
            ]
            return AkuUnion(str, args)
        raise TypeError

    def add_argument(self, argument_parser: ArgumentParser, name: str, default: Any,
                     prefixes: Tuple[str, ...], domain: Tuple[str, ...]) -> None:
        choices = {c.__name__: c for c in self.choices}

        class UnionAction(Action):
            def __call__(self, parser: ArgumentParser, namespace: Namespace, values, option_string=None):
                setattr(namespace, self.dest, choices[values])
                self.required = False

                num_actions = len(parser._actions)
                AkuFn(choices[values], None).add_argument(
                    argument_parser=parser, name=name,
                    prefixes=prefixes, domain=domain, default=None,
                )
                parser._actions, new_actions = parser._actions[:num_actions], parser._actions[num_actions:]
                setattr(parser, NEW_ACTIONS, getattr(parser, NEW_ACTIONS, []) + new_actions)

        prefixes_name = join_names(prefixes, name)
        argument_parser.add_argument(
            f'--{prefixes_name}', dest=join_dests(domain + (name,), '@fn'), help=prefixes_name,
            type=self.tp, choices=tuple(choices.keys()), required=True, default=SUPPRESS,
            action=UnionAction, metavar=f'{{{", ".join(choices.keys())}}}[fn]'
        )


class Aku(ArgumentParser):
    def __init__(self, prog=__file__,
                 usage=None,
                 description=None,
                 epilog=None,
                 parents=(),
                 formatter_class=functools.partial(
                     ArgumentDefaultsHelpFormatter,
                     max_help_position=get_max_help_position(),
                 ),
                 prefix_chars='-',
                 fromfile_prefix_chars=None,
                 argument_default=None,
                 conflict_handler='error',
                 add_help=True,
                 allow_abbrev=True,
                 exit_on_error=True) -> None:
        super(Aku, self).__init__(
            prog, usage, description, epilog, parents, formatter_class, prefix_chars,
            fromfile_prefix_chars, argument_default, conflict_handler, add_help, allow_abbrev,
            exit_on_error,
        )
        _init_argument_parser(self)

        self._functions = []

    def option(self, fn):
        self._functions.append(Type[fn])
        return fn

    def parse_args(self, args=None) -> Namespace:
        AkuTp[Union[tuple(self._functions)]].add_argument(
            self, name='root', default=SUPPRESS,
            prefixes=(), domain=(),
        )

        namespace, args = None, None
        while True:
            namespace, args = self.parse_known_args(args=args, namespace=namespace)
            if hasattr(self, NEW_ACTIONS):
                self._actions = self._actions + getattr(self, NEW_ACTIONS)
                delattr(self, NEW_ACTIONS)
            else:
                break

        return namespace

    def run(self, namespace: Namespace = None):
        if namespace is None:
            namespace = self.parse_args()
        if isinstance(namespace, Namespace):
            namespace = namespace.__dict__

        args = {}
        for key, value in namespace.items():
            collection = args
            *names, key = key.split('.')
            for name in names:
                collection = collection.setdefault(name, {})
            if key == '@fn':
                collection[key] = value
            else:
                collection.setdefault('@args', {})[key] = value

        def recur(x):
            if isinstance(x, dict):
                if '@fn' in x:
                    kwargs = {key: recur(value) for key, value in x['@args'].items()}
                    return functools.partial(x['@fn'], **kwargs)
                else:
                    return {
                        key: recur(value)
                        for key, value in x.items()
                    }
            else:
                return x

        ret = recur(args)
        assert len(ret) == 1
        for _, fn in ret.items():
            if inspect.getfullargspec(fn).varkw is None:
                return fn()
            else:
                return fn(**{'@aku': args})
