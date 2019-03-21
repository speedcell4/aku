import typing

from aku.analysis import Aku

aku = Aku()


@aku.register
def add(a: int, b: float):
    print(f'a => {a}')
    print(f'b => {b}')


@aku.register
def sub(x: int, y: float = None):
    print(f'x => {x}')
    print(f'y => {y}')


@aku.register
def mul(g: typing.Union[int, str] = 2, h: bool = False):
    print(f'g => {g}')
    print(f'h => {h}')


@aku.register
def div(w: typing.List[int] = [1, 2], k: typing.Tuple[float, ...] = (2.0, 3.0)):
    print(f'w => {w}')
    print(f'k => {k}')


aku.run()
