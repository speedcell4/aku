import typing

from aku.analysis import Aku

aku = Aku()


@aku.register
def foo(a: int, b: float):
    print(f'a => {a}')
    print(f'b => {b}')


@aku.register
def bar(x: int, y: float = None):
    print(f'x => {x}')
    print(f'y => {y}')


@aku.register
def baz(g: typing.Union[int, str] = 2, h: bool = False):
    print(f'g => {g}')
    print(f'h => {h}')


@aku.register
def qux(w: typing.List[int] = [1, 2], k: typing.Tuple[float, ...] = (2.0, 3.0)):
    print(f'w => {w}')
    print(f'k => {k}')


@aku.register
def quux(r: (1, 2, 3) = 3):
    print(f'r => {r}')


def add(x: int, y: int):
    return x + y


def sub(x: float, y: float):
    return x - y


@aku.register
def corge(name: str = 'aku', cal: (add, sub) = add):
    print(f'hello {name}')
    print(f'cal() => {cal()}')


aku.run()
