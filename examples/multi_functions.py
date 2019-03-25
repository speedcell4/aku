import typing
from typing import Type, Union

from aku.analysis import Aku

aku = Aku()


@aku.register
def foo(a: int = 1, b: float = 2, c: complex = 3 + 4j, d: bool = True, e: str = 'e :: string'):
    print(f'a => {a}')
    print(f'b => {b}')
    print(f'c => {c}')
    print(f'd => {d}')
    print(f'e => {e}')


@aku.register
def bar(a: int = None, b: float = None, c: complex = None, d: bool = None, e: str = None):
    print(f'a => {a}')
    print(f'b => {b}')
    print(f'c => {c}')
    print(f'd => {d}')
    print(f'e => {e}')


@aku.register
def baz(a: Union[int, str] = 2, b: Union[float, float] = 3.0, c: Union[int, float] = None):
    print(f'a => {a}')
    print(f'b => {b}')
    print(f'c => {c}')


@aku.register
def qux(a: typing.List[int] = [6, 7], b: typing.Tuple[float, ...] = (8.0, 9.0)):
    print(f'a => {a}')
    print(f'b => {b}')


@aku.register
def quux(a: (1, 2, 3) = 3):
    print(f'a => {a}')


def add(x: int, y: int):
    return x + y


def sub(x: float, y: float):
    return x - y


@aku.register
def corge(cal: Type[Union[add, sub]] = add):
    print(f'cal() => {cal()}')


aku.run()
