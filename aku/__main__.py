from argparse import ArgumentDefaultsHelpFormatter, SUPPRESS
from typing import Type
from typing import Union, List, Literal

from aku.tp import Aku, _init_argument_parser, AkuTp

if __name__ == '__main__':
    aku = Aku(
        prog='oar', usage=None, description=None,
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    _init_argument_parser(aku)


    def foo(x: int = 3, y: str = '4', **kwargs):
        for _ in range(x):
            print(f'foo => {y}')
        print(kwargs['@aku'])


    def bar(x: Literal[1, 2, 3] = 2, y: List[int] = [2, 3, 4], **kwargs):
        print(f'bar.x => {x}')
        print(f'bar.y => {y}')
        print(kwargs['@aku'])


    AkuTp[Type[Union[foo, bar]]].add_argument(
        argument_parser=aku, name='fn_', default=SUPPRESS,
        prefixes=(), domain=(),
    )

    print(aku.run())
