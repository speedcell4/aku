import functools
from argparse import ArgumentParser, Action, Namespace, SUPPRESS
from typing import Union, Tuple, Any

from aku.actions import StoreAction, AppendListAction
from aku.compat import Literal, get_origin, get_args
from aku.utils import register_homo_tuple, register_hetero_tuple, tp_iter, join_names, join_dests, AKU_FN, \
    get_action_group

delay = []


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

    def __class_getitem__(cls, tp) -> 'AkuTp':
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
            type=self.tp, choices=self.choices, required=None if default == SUPPRESS else False,
            action=StoreAction, default=default, metavar=self.tp.__name__.lower(),
        )


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
            type=self.tp, choices=self.choices, required=None if default == SUPPRESS else False,
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
            type=register_homo_tuple(self.tp, argument_parser), choices=self.choices,
            required=None if default == SUPPRESS else False,
            action=StoreAction, default=default, metavar=f'({self.tp.__name__.lower()}, ...)',
        )


class AkuHeteroTuple(AkuTp):
    __class_getitem__ = AkuHomoTuple.__class_getitem__

    def add_argument(self, argument_parser: ArgumentParser, name: str, default: Any,
                     prefixes: Tuple[str, ...], domain: Tuple[str, ...]) -> None:
        metavars = ', '.join(t.__name__.lower() for t in self.tp)
        prefixes_name = join_names(prefixes, name)
        argument_parser.add_argument(
            f'--{prefixes_name}', dest=join_dests(domain, name), help=prefixes_name,
            type=register_hetero_tuple(self.tp, argument_parser), choices=self.choices,
            required=None if default == SUPPRESS else False,
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
            type=self.tp, choices=self.choices, required=None if default == SUPPRESS else False,
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
        elif origin is Union:
            args = [
                get_args(arg)[0]
                for arg in get_args(tp)
            ]
            return AkuUnion(str, args)
        raise TypeError

    def add_argument(self, argument_parser: ArgumentParser, name: str, default: Any,
                     prefixes: Tuple[str, ...], domain: Tuple[str, ...]) -> None:

        if name is not None:
            domain = domain + (name,)
            if name.endswith('_'):
                _, argument_parser = get_action_group(argument_parser, join_names(prefixes, name))
                prefixes = prefixes + (name[:-1],)

        argument_parser.set_defaults(**{join_dests(domain, AKU_FN): (self.tp, name)})

        for arg, tp, df in tp_iter(self.tp):
            AkuTp[tp].add_argument(
                argument_parser=argument_parser, name=arg,
                prefixes=prefixes, domain=domain, default=df,
            )


class AkuUnion(AkuTp):
    __class_getitem__ = AkuFn.__class_getitem__

    def add_argument(self, argument_parser: ArgumentParser, name: str, default: Any,
                     prefixes: Tuple[str, ...], domain: Tuple[str, ...]) -> None:
        choices = {c.__name__: c for c in self.choices}

        class UnionAction(Action):
            def __call__(self, parser: ArgumentParser, namespace: Namespace, values, option_string=None):
                setattr(namespace, self.dest, (choices[values], values))
                self.required = False

                delay.append(functools.partial(
                    AkuFn(choices[values], None).add_argument,
                    argument_parser=parser, name=name,
                    prefixes=prefixes, domain=domain, default=None,
                ))

        prefixes_name = join_names(prefixes, name)
        argument_parser.add_argument(
            f'--{prefixes_name}', dest=join_dests(domain + (name,), AKU_FN), help=prefixes_name,
            type=self.tp, choices=tuple(choices.keys()), required=None, default=SUPPRESS,
            action=UnionAction, metavar=f'fn{{{", ".join(choices.keys())}}}'
        )
