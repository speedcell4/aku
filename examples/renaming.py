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


def fred(_: Type[Foo.quux] = Foo.quux, *args, **kwargs):
    _(*args, **kwargs)


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


def thud(_: Type[Bar.quux] = Bar.quux, *args, **kwargs):
    _(*args, **kwargs)


@app.register
def main(a: Union[Type[foo], Type[bar]] = foo,
         b: Union[Type[Foo], Type[Bar]] = Foo,
         c: Union[Type[Foo.qux], Type[Bar.qux]] = Foo.qux,
         d: Union[Type[fred], Type[thud]] = fred, **kwargs):
    a()
    b()
    c()
    d()
    print(kwargs['@aku'])


if __name__ == '__main__':
    app.run()
