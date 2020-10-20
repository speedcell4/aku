from argparse import SUPPRESS
from pathlib import Path
from typing import Type, Tuple
from typing import Union, List, Literal

from aku.tp import Aku, AkuTp

if __name__ == '__main__':
    aku = Aku()


    def foo(x: int, y: str = '4', z: bool = True, w: Path = Path.home(), **kwargs):
        for _ in range(x):
            print(f'foo => {y}')
        print(f'z => {z}')
        print(f'w => {w}')
        print(kwargs['@aku'])


    def bar(x: Literal[1, 2, 3] = 2, y: List[int] = [2, 3, 4],
            z: Tuple[float, ...] = (), w: Tuple[float, str, int] = (1., '2', 3), **kwargs):
        print(f'bar.x => {x}')
        print(f'bar.y => {y}')
        print(f'bar.z => {z}')
        print(f'bar.w => {w}')
        print(kwargs['@aku'])


    AkuTp[Type[Union[foo, bar]]].add_argument(
        argument_parser=aku, name='fn', default=SUPPRESS,
        prefixes=(), domain=(),
    )

    print(aku.run())
