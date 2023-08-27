from typing import Type
from typing import Union

from aku import Aku

aku = Aku()


@aku.register
def empty(x):
    print(f'x => {x}')


def foo(x: int = 1):
    print(f'x => {x}')


def bar(x: int = 2):
    print(f'x => {x}')


def baz(x: int = 3):
    print(f'x => {x}')


@aku.register
def conflicting(fn1: Type[foo] = foo, fn2: Union[Type[bar], Type[baz]] = baz):
    fn1()
    fn2()


if __name__ == '__main__':
    aku.run()
