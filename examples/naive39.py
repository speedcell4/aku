from pathlib import Path
from typing import Literal

from aku import Aku

aku = Aku()


@aku.option
def foo(x: int, y: str = '4', z: bool = True, w: Path = Path.home(), **kwargs):
    print(f'foo.x => {x}')
    print(f'foo.y => {y}')
    print(f'foo.z => {z}')
    print(f'foo.w => {w}')
    print(kwargs['@aku'])


@aku.option
def bar(x: Literal[1, 2, 3] = 2, y: list[int] = [2, 3, 4],
        z: tuple[float, ...] = (), w: tuple[float, str, int] = (1., '2', 3), **kwargs):
    print(f'bar.x => {x}')
    print(f'bar.y => {y}')
    print(f'bar.z => {z}')
    print(f'bar.w => {w}')
    print(kwargs['@aku'])


print(aku.run())
