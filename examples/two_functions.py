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


aku.run()
