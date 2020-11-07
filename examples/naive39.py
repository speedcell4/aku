from pathlib import Path
from typing import Literal

from aku import Aku

aku = Aku()


@aku.option
def foo(x: int, y: str = '4', z: bool = True, w: Path = Path.home(), **kwargs):
    print(f'{foo.__name__}.x => {x}')
    print(f'{foo.__name__}.y => {y}')
    print(f'{foo.__name__}.z => {z}')
    print(f'{foo.__name__}.w => {w}')
    print(f'{foo.__name__}.@aku => {kwargs["@aku"]}')


@aku.option
def bar(x: Literal[1, 2, 3] = 2, y: list[int] = [2, 3, 4],
        z: tuple[float, ...] = (), w: tuple[float, str, int] = (1., '2', 3), **kwargs):
    print(f'{bar.__name__}.x => {x}')
    print(f'{bar.__name__}.y => {y}')
    print(f'{bar.__name__}.z => {z}')
    print(f'{bar.__name__}.w => {w}')
    print(f'{bar.__name__}.@aku => {kwargs["@aku"]}')


print(aku.run())
