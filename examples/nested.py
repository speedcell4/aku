from typing import Union, Type

from aku import Aku


def baz(a: float = 4, b: bool = True):
    print(f'a => {a}')
    print(f'b => {b}')


def qux(c: str = 'nice', d: float = 0.):
    print(f'c => {c}')
    print(f'd => {d}')


class Foo(object):
    def __init__(self, f: Union[Type[baz], Type[qux]], g: int = 1, h: int = 2):
        super(Foo, self).__init__()
        f()
        print(f'g => {g}')
        print(f'h => {h}')


class Bar(object):
    def __init__(self, o: str = 'o', p_with_many_underlines: str = 'p_with_many_underlines'):
        super(Bar, self).__init__()
        print(f'o => {o}')
        print(f'p_with_many_underlines => {p_with_many_underlines}')


aku = Aku(always_use_subparse=True)


@aku.option
def first(x: Union[Type[Foo], Type[Bar]] = Foo, **kwargs):
    print(f'kwargs => {kwargs}')
    x()


@aku.option
def another(x: Union[Type[Foo], Type[Bar]], **kwargs):
    print(f'kwargs => {kwargs}')
    x()


if __name__ == '__main__':
    aku.run()
