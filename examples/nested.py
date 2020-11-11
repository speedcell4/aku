from typing import Union, Type

from aku import Aku


def baz(c: float, d: bool = True):
    print(f'c => {c}')
    print(f'd => {d}')


def qux(e: str = 'nice', f: float = 0.):
    print(f'e => {e}')
    print(f'f => {f}')


class Foo(object):
    def __init__(self, fn: Union[Type[baz], Type[qux]], a: int = 1, b: int = 2):
        super(Foo, self).__init__()
        fn()
        print(f'a => {a}')
        print(f'b => {b}')


class Bar(object):
    def __init__(self, x: str = 'x', y: str = 'y'):
        super(Bar, self).__init__()
        print(f'x => {x}')
        print(f'y => {y}')


aku = Aku()


@aku.option
def main(x: Union[Type[Foo], Type[Bar]]):
    x()


if __name__ == '__main__':
    aku.run()
