from argparse import ArgumentDefaultsHelpFormatter, SUPPRESS
from typing import Type
from typing import Union, List, Tuple, Literal

from aku.tp import Aku, _init_argument_parser, AkuTp

if __name__ == '__main__':
    parser = Aku(
        prog='oar', usage=None, description=None,
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    _init_argument_parser(parser)


    def foo(f1: int = 3, f2: str = '4'):
        print(f'a => {f1}')
        print(f'w => {f2}')


    def bar(b1: Literal[1, 2, 3] = 2, b2: List[int] = [2, 3, 4]):
        print(f'c => {b1}')
        print(f'd => {b2}')


    def baz(z1: Tuple[int, str], z2: Tuple[float, ...]):
        pass


    def nice(a: Type[Union[foo, bar, baz]]):
        pass


    AkuTp[Type[nice]].add_argument(parser, 'nice', SUPPRESS)

    print(parser.parse_args())
