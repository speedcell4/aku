import inspect
import re
from argparse import ArgumentParser, Action, Namespace, SUPPRESS
from re import Pattern
from typing import Union, Tuple, Literal, Any
from typing import get_origin, get_args, get_type_hints

NEW_ACTIONS = '_new_actions'


def tp_none(arg_strings: str) -> type(None):
    arg_strings = arg_strings.lower().strip()
    if arg_strings in ('nil', 'null', 'none'):
        return None
    raise ValueError


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

    def add_argument(self, argument_parser: ArgumentParser, name: str, default: Any) -> None:
        raise NotImplementedError


class StorePrimitiveAction(Action):
    def __call__(self, parser: ArgumentParser, namespace: Namespace, values, option_string=None):
        setattr(namespace, self.dest, values)
        self.required = False


class AkuPrimitive(AkuTp):
    def __class_getitem__(cls, tp):
        tp, origin, args = tp
        if origin is None:
            return AkuPrimitive(tp, None)
        raise TypeError

    def add_argument(self, argument_parser: ArgumentParser, name: str, default: Any) -> None:
        argument_parser.add_argument(
            f'--{name}', type=self.tp, choices=self.choices, required=True,
            action=StorePrimitiveAction, default=default,
        )


class AppendListAction(Action):
    def __call__(self, parser: ArgumentParser, namespace: Namespace, values, option_string=None):
        flag_name = '_aku_visited'
        if not getattr(self, flag_name, False):
            setattr(namespace, self.dest, [])
            setattr(self, flag_name, True)
        getattr(namespace, self.dest).append(values)


class AkuList(AkuTp):
    def __class_getitem__(cls, tp):
        tp, origin, args = tp
        if origin is list:
            return AkuList(args[0], None)
        raise TypeError

    def add_argument(self, argument_parser: ArgumentParser, name: str, default: Any) -> None:
        argument_parser.add_argument(
            f'--{name}', type=self.tp, choices=self.choices, required=True,
            action=AppendListAction, default=default,
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

    def add_argument(self, argument_parser: ArgumentParser, name: str, default: Any) -> None:
        argument_parser.add_argument(
            f'--{name}', type=register_homo_tuple(self.tp, argument_parser), choices=self.choices, required=True,
            action=StorePrimitiveAction, default=default,
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

    def add_argument(self, argument_parser: ArgumentParser, name: str, default: Any) -> None:
        argument_parser.add_argument(
            f'--{name}', type=register_hetero_tuple(self.tp, argument_parser), choices=self.choices, required=True,
            action=StorePrimitiveAction, default=default,
        )


class AkuLiteral(AkuTp):
    def __class_getitem__(cls, tp):
        tp, origin, args = tp
        if origin is Literal:
            if len(args) > 0:
                tp = type(args[0])
                for arg in args[1:]:
                    assert isinstance(arg, tp), f'{type(arg)} is not {tp}'
                return AkuLiteral(tp, args)
        raise TypeError

    def add_argument(self, argument_parser: ArgumentParser, name: str, default: Any) -> None:
        argument_parser.add_argument(
            f'--{name}', type=self.tp, choices=self.choices, required=True,
            action=StorePrimitiveAction, default=default,
        )


class AkuType(AkuTp):
    def __class_getitem__(cls, tp):
        tp, origin, args = tp
        if origin is type:
            if len(args) == 1:
                if get_origin(args[0]) == Union:
                    return AkuUnion(str, get_args(args[0]))
                else:
                    return AkuType(args[0], None)
        raise TypeError

    def add_argument(self, argument_parser: ArgumentParser, name: str, default: Any) -> None:
        for arg, tp, df in tp_iter(self.tp):
            tp = AkuTp[tp]
            if name.endswith('_'):
                arg = f'{name}{arg}'
            tp.add_argument(argument_parser=argument_parser, name=arg, default=df)


class AkuUnion(AkuTp):
    def __class_getitem__(cls, tp):
        tp, origin, args = tp
        if origin is type:
            if len(args) == 1:
                if get_origin(args[0]) == Union:
                    return AkuUnion(str, get_args(args[0]))
                else:
                    return AkuType(args[0], None)
        elif origin is Union:
            args = [
                get_args(arg)[0]
                for arg in get_args(tp)
            ]
            return AkuUnion(str, args)
        raise TypeError

    def add_argument(self, argument_parser: ArgumentParser, name: str, default: Any) -> None:
        choices = {c.__name__: c for c in self.choices}

        class UnionAction(Action):
            def __call__(self, parser: ArgumentParser, namespace: Namespace, values, option_string=None):
                setattr(namespace, self.dest, values)
                self.required = False

                num_actions = len(parser._actions)
                AkuType(choices[values], None).add_argument(argument_parser=parser, name=name, default=None)
                parser._actions, new_actions = parser._actions[:num_actions], parser._actions[num_actions:]
                setattr(parser, NEW_ACTIONS, getattr(parser, NEW_ACTIONS, []) + new_actions)

        argument_parser.add_argument(
            f'--{name}' if not name.endswith('_') else f'--{name[:-1]}',
            type=self.tp, choices=tuple(choices.keys()), required=True,
            action=UnionAction,
        )


class Aku(ArgumentParser):
    def parse_args(self, args=None) -> Namespace:
        namespace, args = None, None
        while True:
            namespace, args = self.parse_known_args(args=args, namespace=namespace)
            if hasattr(self, NEW_ACTIONS):
                self._actions = self._actions + getattr(self, NEW_ACTIONS)
                delattr(self, NEW_ACTIONS)
            else:
                break

        return namespace
