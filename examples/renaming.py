from typing import Type, Union

from aku import Aku

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
    def quux(cls, w: str = 'w') -> None:
        print(f'w => {w}')


@app.register
def main(a: Union[Type[foo], Type[bar]] = foo,
         b: Union[Type[Foo], Type[Bar]] = Foo,
         c: Union[Type[Foo.qux], Type[Bar.qux]] = Foo.qux, **kwargs):
    a()
    b()
    c()
    print(kwargs['@aku'])


if __name__ == '__main__':
    app.run()
