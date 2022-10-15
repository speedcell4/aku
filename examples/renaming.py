from typing import Type, Union

from aku import Aku
from aku.utils import rename

app = Aku()


def foo(x: str = 'x'):
    print(f'x => {x}')


def bar(x: str = 'x'):
    print(f'x => {x}')


class Foo(object):
    def __init__(self, y: str = 'y') -> None:
        super(Foo, self).__init__()
        print(f'y => {y}')

    @classmethod
    def qux(cls, z: str = 'z') -> None:
        print(f'z => {z}')

    @classmethod
    @rename('thud')
    def quux(cls, w: str = 'w') -> None:
        print(f'w => {w}')


class Bar(object):
    def __init__(self, y: str = 'y') -> None:
        super(Bar, self).__init__()
        print(f'y => {y}')

    @classmethod
    def qux(cls, z: str = 'z') -> None:
        print(f'z => {z}')

    @classmethod
    @rename('fred')
    def quux(cls, w: str = 'w') -> None:
        print(f'w => {w}')


@app.register
def main(fn: Union[Type[foo], Type[bar]] = foo,
         cls: Union[Type[Foo], Type[Bar]] = Foo,
         method1: Union[Type[Foo.qux], Type[Bar.qux]] = Foo.qux,
         method2: Union[Type[Foo.quux], Type[Bar.quux]] = Foo.quux):
    fn()
    cls()
    method1()
    method2()


if __name__ == '__main__':
    app.run()
