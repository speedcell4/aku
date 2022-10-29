import functools
from argparse import ArgumentParser, Action, Namespace, SUPPRESS
from typing import Union, Tuple, Any, Literal, get_origin, get_args, Type

from aku.actions import StoreAction, AppendListAction
from aku.utils import AKU_FN, AKU_DELAY, get_action_group
from aku.utils import get_name, get_dest, get_option, iter_annotations
from aku.utils import register_homo_tuple_type, register_hetero_tuple_type
from aku.utils import register_set_type, register_frozenset_type


class AkuTp(object):
    def __init__(self, tp: Type, choices: Any = None) -> None:
        super(AkuTp, self).__init__()

        name = get_name(tp=tp)
        if hasattr(tp, '__supertype__'):
            if get_origin(tp) is None and get_args(tp) == ():
                name = tp.__name__
                tp = tp.__supertype__

        self.tp = tp
        self.name = name
        self.choices = choices

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.name}, {self.choices})'

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

    def add_argument(self, argument_parser: ArgumentParser, name: str, default: Any, domain: Tuple[str, ...]) -> None:
        raise NotImplementedError


class AkuPrimitive(AkuTp):
    def __class_getitem__(cls, tp):
        t, origin, _ = tp
        if origin is None:
            return AkuPrimitive(t)
        raise TypeError

    def add_argument(self, argument_parser: ArgumentParser, name: str, default: Any, domain: Tuple[str, ...]) -> None:
        option = get_option(domain, name)
        argument_parser.add_argument(
            f'--{option}', dest=get_dest(domain, name), help=option,
            type=self.tp, choices=self.choices, required=None if default == SUPPRESS else False,
            action=StoreAction, default=default, metavar=self.name,
        )


class AkuList(AkuTp):
    def __class_getitem__(cls, tp):
        _, origin, (t, *_) = tp
        if origin is list:
            return AkuList(t)
        raise TypeError

    def add_argument(self, argument_parser: ArgumentParser, name: str, default: Any, domain: Tuple[str, ...]) -> None:
        option = get_option(domain, name)
        argument_parser.add_argument(
            f'--{option}', dest=get_dest(domain, name), help=option,
            type=self.tp, choices=self.choices, required=None if default == SUPPRESS else False,
            action=AppendListAction, default=default, metavar=f'[{self.name}]',
        )


class AkuHomoTuple(AkuTp):
    def __class_getitem__(cls, tp):
        _, origin, (t, *tps) = tp
        if origin is tuple:
            if len(tps) == 1 and tps[0] is ...:
                return AkuHomoTuple(t)
            else:
                return AkuHeteroTuple((t, *tps))
        raise TypeError

    def add_argument(self, argument_parser: ArgumentParser, name: str, default: Any, domain: Tuple[str, ...]) -> None:
        option = get_option(domain, name)
        argument_parser.add_argument(
            f'--{option}', dest=get_dest(domain, name), help=option,
            type=register_homo_tuple_type(self.tp, argument_parser), choices=self.choices,
            required=None if default == SUPPRESS else False,
            action=StoreAction, default=default, metavar=f'({self.name}, ...)',
        )


class AkuHeteroTuple(AkuTp):
    __class_getitem__ = AkuHomoTuple.__class_getitem__

    def add_argument(self, argument_parser: ArgumentParser, name: str, default: Any, domain: Tuple[str, ...]) -> None:
        option = get_option(domain, name)
        argument_parser.add_argument(
            f'--{option}', dest=get_dest(domain, name), help=option,
            type=register_hetero_tuple_type(self.tp, argument_parser), choices=self.choices,
            required=None if default == SUPPRESS else False,
            action=StoreAction, default=default, metavar=f"({', '.join(AkuTp[t].name for t in self.tp)})",
        )


class AkuSet(AkuTp):
    def __class_getitem__(cls, tp):
        _, origin, (t, *_) = tp
        if origin is set:
            return AkuSet(t)
        raise TypeError

    def add_argument(self, argument_parser: ArgumentParser, name: str, default: Any, domain: Tuple[str, ...]) -> None:
        option = get_option(domain, name)
        argument_parser.add_argument(
            f'--{option}', dest=get_dest(domain, name), help=option,
            type=register_set_type(self.tp, argument_parser), choices=self.choices,
            required=None if default == SUPPRESS else False,
            action=StoreAction, default=default, metavar=f'{{{self.name}}}',
        )


class AkuFrozenSet(AkuTp):
    def __class_getitem__(cls, tp):
        _, origin, (t, *_) = tp
        if origin is frozenset:
            return AkuFrozenSet(t)
        raise TypeError

    def add_argument(self, argument_parser: ArgumentParser, name: str, default: Any, domain: Tuple[str, ...]) -> None:
        option = get_option(domain, name)
        argument_parser.add_argument(
            f'--{option}', dest=get_dest(domain, name), help=option,
            type=register_frozenset_type(self.tp, argument_parser), choices=self.choices,
            required=None if default == SUPPRESS else False,
            action=StoreAction, default=default, metavar=f'frozenset{{{self.name}}}',
        )


class AkuLiteral(AkuTp):
    def __class_getitem__(cls, tp):
        _, origin, (t, *tps) = tp
        if origin is Literal:
            tp = type(t)
            for t in (t, *tps):
                assert get_origin(t) is None, f'{t} is not a primitive type'
                assert isinstance(t, tp), f'{type(t)} is not {tp}'
            return AkuLiteral(tp, choices=(t, *tps))
        raise TypeError

    def add_argument(self, argument_parser: ArgumentParser, name: str, default: Any, domain: Tuple[str, ...]) -> None:
        option = get_option(domain, name)
        argument_parser.add_argument(
            f'--{option}', dest=get_dest(domain, name), help=option,
            type=self.tp, choices=self.choices, required=None if default == SUPPRESS else False,
            action=StoreAction, default=default, metavar=f'{self.name}{set(self.choices)}',
        )


class AkuFn(AkuTp):
    def __class_getitem__(cls, tp):
        _, origin, (t, *tps) = tp
        if origin is type:
            if len(tps) == 0:
                if get_origin(t) == Union:
                    return AkuUnion(str, choices=get_args(t))
                else:
                    return AkuFn(t)
        elif origin is Union:
            return AkuUnion(str, choices=[get_args(t)[0] for t in (t, *tps)])
        raise TypeError

    def add_argument(self, argument_parser: ArgumentParser, name: str, default: Any, domain: Tuple[str, ...]) -> None:
        if name is not None:
            if name.endswith('_'):
                _, argument_parser = get_action_group(argument_parser, get_option(domain, name))
            domain = domain + (name,)

        if argument_parser.get_default(get_dest(domain, AKU_FN)) is None:
            argument_parser.set_defaults(**{get_dest(domain, AKU_FN): (self.tp, self.name)})

        for name, tp, default in iter_annotations(self.tp):
            AkuTp[tp].add_argument(
                argument_parser=argument_parser, name=name,
                domain=domain, default=default,
            )


class AkuUnion(AkuTp):
    __class_getitem__ = AkuFn.__class_getitem__

    def add_argument(self, argument_parser: ArgumentParser, name: str, default: Any, domain: Tuple[str, ...]) -> None:
        choices = [AkuTp[c] for c in self.choices]
        choices = {c.name: c.tp for c in choices}

        class UnionAction(Action):
            def __call__(self, parser: ArgumentParser, namespace: Namespace, values, option_string=None):
                setattr(namespace, self.dest, (choices[values], values))
                self.required = False

                parser.register(AKU_DELAY, domain + (name,), functools.partial(
                    AkuTp[Type[choices[values]]].add_argument,
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
            default_tp = AkuTp[Type[default]]
            argument_parser.register(AKU_DELAY, domain + (name,), functools.partial(
                default_tp.add_argument,
                argument_parser=argument_parser, name=name,
                domain=domain, default=None,
            ))

            argument_parser.add_argument(
                f'--{option}', dest=get_dest(domain + (name,), AKU_FN), help=option,
                type=self.tp, choices=tuple(choices.keys()), required=False, default=(default_tp.tp, default_tp.name),
                action=UnionAction, metavar=f'fn{{{", ".join(choices.keys())}}}'
            )
