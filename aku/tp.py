import functools
from argparse import ArgumentParser, Action, Namespace, SUPPRESS
from typing import Union, Tuple, Any

from aku.actions import StoreAction, AppendListAction
from aku.compat import Literal, get_origin, get_args
from aku.utils import AKU_FN, AKU_DELAY, get_action_group, register_set_type, register_frozenset_type, get_name, \
    get_option
from aku.utils import register_homo_tuple_type, register_hetero_tuple, iter_annotations, get_dest


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
        for aku_tp in cls.registry:
            try:
                return aku_tp[tp, origin, args]
            except TypeError:
                pass
        raise TypeError(f'unsupported annotation {tp}')

    def add_argument(self, argument_parser: ArgumentParser, name: str, default: Any,
                     domain: Tuple[str, ...]) -> None:
        raise NotImplementedError


class AkuPrimitive(AkuTp):
    def __class_getitem__(cls, tp):
        tp, origin, args = tp
        if origin is None:
            return AkuPrimitive(tp, None)
        raise TypeError

    def add_argument(self, argument_parser: ArgumentParser, name: str, default: Any, domain: Tuple[str, ...]) -> None:
        option = get_option(domain, name)
        argument_parser.add_argument(
            f'--{option}', dest=get_dest(domain, name), help=option,
            type=self.tp, choices=self.choices, required=None if default == SUPPRESS else False,
            action=StoreAction, default=default, metavar=get_name(self.tp).lower(),
        )


class AkuList(AkuTp):
    def __class_getitem__(cls, tp):
        tp, origin, args = tp
        if origin is list:
            return AkuList(args[0], None)
        raise TypeError

    def add_argument(self, argument_parser: ArgumentParser, name: str, default: Any, domain: Tuple[str, ...]) -> None:
        option = get_option(domain, name)
        argument_parser.add_argument(
            f'--{option}', dest=get_dest(domain, name), help=option,
            type=self.tp, choices=self.choices, required=None if default == SUPPRESS else False,
            action=AppendListAction, default=default, metavar=f'[{get_name(self.tp).lower()}]',
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

    def add_argument(self, argument_parser: ArgumentParser, name: str, default: Any, domain: Tuple[str, ...]) -> None:
        option = get_option(domain, name)
        argument_parser.add_argument(
            f'--{option}', dest=get_dest(domain, name), help=option,
            type=register_homo_tuple_type(self.tp, argument_parser), choices=self.choices,
            required=None if default == SUPPRESS else False,
            action=StoreAction, default=default, metavar=f'({get_name(self.tp).lower()}, ...)',
        )


class AkuHeteroTuple(AkuTp):
    __class_getitem__ = AkuHomoTuple.__class_getitem__

    def add_argument(self, argument_parser: ArgumentParser, name: str, default: Any, domain: Tuple[str, ...]) -> None:
        option = get_option(domain, name)
        argument_parser.add_argument(
            f'--{option}', dest=get_dest(domain, name), help=option,
            type=register_hetero_tuple(self.tp, argument_parser), choices=self.choices,
            required=None if default == SUPPRESS else False,
            action=StoreAction, default=default, metavar=f"({', '.join(get_name(t).lower() for t in self.tp)})",
        )


class AkuSet(AkuTp):
    def __class_getitem__(cls, tp):
        tp, origin, args = tp
        if origin is set:
            return AkuSet(args[0], None)
        raise TypeError

    def add_argument(self, argument_parser: ArgumentParser, name: str, default: Any, domain: Tuple[str, ...]) -> None:
        option = get_option(domain, name)
        argument_parser.add_argument(
            f'--{option}', dest=get_dest(domain, name), help=option,
            type=register_set_type(self.tp, argument_parser), choices=self.choices,
            required=None if default == SUPPRESS else False,
            action=StoreAction, default=default, metavar=f'{{{get_name(self.tp).lower()}}}',
        )


class AkuFrozenSet(AkuTp):
    def __class_getitem__(cls, tp):
        tp, origin, args = tp
        if origin is frozenset:
            return AkuFrozenSet(args[0], None)
        raise TypeError

    def add_argument(self, argument_parser: ArgumentParser, name: str, default: Any, domain: Tuple[str, ...]) -> None:
        option = get_option(domain, name)
        argument_parser.add_argument(
            f'--{option}', dest=get_dest(domain, name), help=option,
            type=register_frozenset_type(self.tp, argument_parser), choices=self.choices,
            required=None if default == SUPPRESS else False,
            action=StoreAction, default=default, metavar=f'frozenset{{{get_name(self.tp).lower()}}}',
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

    def add_argument(self, argument_parser: ArgumentParser, name: str, default: Any, domain: Tuple[str, ...]) -> None:
        option = get_option(domain, name)
        argument_parser.add_argument(
            f'--{option}', dest=get_dest(domain, name), help=option,
            type=self.tp, choices=self.choices, required=None if default == SUPPRESS else False,
            action=StoreAction, default=default, metavar=f'{get_name(self.tp).lower()}{set(self.choices)}',
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

    def add_argument(self, argument_parser: ArgumentParser, name: str, default: Any, domain: Tuple[str, ...]) -> None:

        if name is not None:
            domain = domain + (name,)
            if name.endswith('_'):
                _, argument_parser = get_action_group(argument_parser, get_option(domain, name))

        argument_parser.set_defaults(**{get_dest(domain, AKU_FN): (self.tp, get_name(self.tp))})

        for arg, tp, df in iter_annotations(self.tp):
            AkuTp[tp].add_argument(
                argument_parser=argument_parser, name=arg,
                domain=domain, default=df,
            )


class AkuUnion(AkuTp):
    __class_getitem__ = AkuFn.__class_getitem__

    def add_argument(self, argument_parser: ArgumentParser, name: str, default: Any, domain: Tuple[str, ...]) -> None:
        choices = {get_name(c): c for c in self.choices}

        class UnionAction(Action):
            def __call__(self, parser: ArgumentParser, namespace: Namespace, values, option_string=None):
                setattr(namespace, self.dest, (choices[values], values))
                self.required = False

                parser.register(AKU_DELAY, domain + (name,), functools.partial(
                    AkuFn(choices[values], None).add_argument,
                    argument_parser=parser, name=name,
                    domain=domain, default=None,
                ))

        option = get_option(domain, name)

        if default == SUPPRESS:
            argument_parser.add_argument(
                f'--{option}', dest=get_dest(domain + (name,), AKU_FN), help=option,
                type=self.tp, choices=tuple(choices.keys()), required=None, default=SUPPRESS,
                action=UnionAction, metavar=f'fn{{{", ".join(choices.keys())}}}'
            )
        else:
            argument_parser.register(AKU_DELAY, domain + (name,), functools.partial(
                AkuFn(default, None).add_argument,
                argument_parser=argument_parser, name=name,
                domain=domain, default=None,
            ))

            argument_parser.add_argument(
                f'--{option}', dest=get_dest(domain + (name,), AKU_FN), help=option,
                type=self.tp, choices=tuple(choices.keys()), required=False, default=(default, default.__name__),
                action=UnionAction, metavar=f'fn{{{", ".join(choices.keys())}}}'
            )
