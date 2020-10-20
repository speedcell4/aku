from argparse import ArgumentDefaultsHelpFormatter, SUPPRESS
from typing import Type
from typing import Union, List, Literal

from aku.tp import Aku, _init_argument_parser, AkuTp

if __name__ == '__main__':
    parser = Aku(
        prog='oar', usage=None, description=None,
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    _init_argument_parser(parser)


    def foo(x: int = 3, y: str = '4'):
        print(f'x => {x}')
        print(f'y => {y}')


    def bar(x: Literal[1, 2, 3] = 2, y: List[int] = [2, 3, 4]):
        print(f'x => {x}')
        print(f'y => {y}')


    AkuTp[Type[Union[foo, bar]]].add_argument(
        argument_parser=parser, name='fn_', default=SUPPRESS,
        prefixes=(), domain=(),
    )

    print(parser.parse_args())
