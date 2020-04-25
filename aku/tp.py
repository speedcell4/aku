import re
from abc import ABCMeta, abstractmethod
from argparse import ArgumentParser
from typing import get_args, get_origin, Any, Union, Literal

from aku.parse_fn import get_parse_fn

COMMA = re.compile(r'\s*,\s*')


class Tp(object, metaclass=ABCMeta):
    def __init__(self, origin, *args: Union['Tp', Any], **kwargs: ['Tp', Any]) -> None:
        super(Tp, self).__init__()
        self.origin = origin
        self.args = args
        self.kwargs = kwargs

    def __class_getitem__(cls, tp):
        args = get_args(tp)
        origin = get_origin(tp)

        if origin is None and args == ():
            return PrimitiveTp(tp)

        if origin is Literal:
            tp = type(args[0])

            assert all(isinstance(a, tp) for a in args), \
                f'all arguments should have the same type {tp.__name__}'
            return PrimitiveTp(tp, *set(args))

        if origin is list and len(args) == 1:
            return ListTp(origin, cls[args[0]])

        if origin is tuple:
            if len(args) == 2 and args[1] is ...:
                return HomoTupleTp(origin, cls[args[0]])
            else:
                return HeteroTupleTp(origin, *[cls[a] for a in args])

        if origin is set:
            return SetTp(origin, cls[args[0]])

        if origin is frozenset:
            return FrozenSetTp(origin, cls[args[0]])

        raise NotImplementedError(f'unsupported {cls.__name__} {tp}')

    @property
    @abstractmethod
    def metavar(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def parse_fn(self, option_string: str) -> Any:
        raise NotImplementedError

    def add_argument(self, argument_parser: ArgumentParser, name: str, default: Any):
        return argument_parser.add_argument(
            f'--{name}', help=f'{name}',
            type=self.parse_fn, metavar=self.metavar, default=repr(default),
        )


class PrimitiveTp(Tp):
    @property
    def metavar(self) -> str:
        if len(self.args) == 0:
            return f'{self.origin.__name__.lower()}'
        return f"{{{', '.join([f'{repr(a)}' for a in self.args])}}}"

    def parse_fn(self, option_string: str) -> Any:
        fn = get_parse_fn(self.origin)
        return fn(option_string.strip())

    def add_argument(self, argument_parser: ArgumentParser, name: str, default: Any):
        return argument_parser.add_argument(
            f'--{name}', help=f'{name}',
            type=self.parse_fn, metavar=self.metavar, default=repr(default),
            choices=self.args if len(self.args) > 0 else None,
        )


class ListTp(Tp):
    @property
    def metavar(self) -> str:
        return f'[{self.args[0].metavar}]'

    def parse_fn(self, option_string: str) -> Any:
        option_string = option_string.strip()
        if not option_string.startswith('[') or not option_string.endswith(']'):
            raise ValueError(f'{option_string} is not a(n) {self.origin.__name__}')

        option_strings = re.split(COMMA, option_string[1:-1])
        return self.origin(self.args[0].parse_fn(s) for s in option_strings)


class HomoTupleTp(Tp):
    @property
    def metavar(self) -> str:
        return f'({self.args[0].metavar}, ...)'

    def parse_fn(self, option_string: str) -> Any:
        option_string = option_string.strip()
        if not option_string.startswith('(') or not option_string.endswith(')'):
            raise ValueError(f'{option_string} is not a(n) {self.origin.__name__}')

        option_strings = re.split(COMMA, option_string[1:-1])
        return self.origin(self.args[0].parse_fn(s) for s in option_strings)


class HeteroTupleTp(Tp):
    @property
    def metavar(self) -> str:
        return f"({', '.join([f'{a.metavar}' for a in self.args])})"

    def parse_fn(self, option_string: str) -> Any:
        option_string = option_string.strip()
        if not option_string.startswith('(') or not option_string.endswith(')'):
            raise ValueError(f'{option_string} is not a(n) {self.origin.__name__}')

        option_strings = re.split(COMMA, option_string[1:-1])
        assert len(option_strings) == len(self.args), \
            f'the number of arguments is not correct, ' \
            f'got {len(option_strings)} but excepted {len(self.args)}'

        return self.origin(a.parse_fn(s) for s, a in zip(option_strings, self.args))


class SetTp(Tp):
    @property
    def metavar(self) -> str:
        return f'{{{self.args[0].metavar}}}'

    def parse_fn(self, option_string: str) -> Any:
        option_string = option_string.strip()
        if not option_string.startswith('{') or not option_string.endswith('}'):
            raise ValueError(f'{option_string} is not a(n) {self.origin.__name__}')

        option_strings = re.split(COMMA, option_string[1:-1])
        return self.origin(self.args[0].parse_fn(s) for s in option_strings)


class FrozenSetTp(Tp):
    @property
    def metavar(self) -> str:
        return f'{{{self.args[0].metavar}}}'

    def parse_fn(self, option_string: str) -> Any:
        option_string = option_string.strip()
        if not option_string.startswith('{') or not option_string.endswith('}'):
            raise ValueError(f'{option_string} is not a(n) {self.origin.__name__}')

        option_strings = re.split(COMMA, option_string[1:-1])
        return self.origin(self.args[0].parse_fn(s) for s in option_strings)
