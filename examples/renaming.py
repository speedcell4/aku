from typing import NewType, Type, Union

from aku import Aku

app = Aku()


class Foo(object):
    def __init__(self, a: int = 1, b: float = 2.0) -> None:
        super(Foo, self).__init__()
        print(f'a => {a}')
        print(f'b => {b}')


class Bar(object):
    def __init__(self, c: int = 3, d: float = 4.0) -> None:
        super(Bar, self).__init__()
        print(f'c => {c}')
        print(f'd => {d}')


class Baz(object):
    def __init__(self, e: int = 3, f: float = 4.0, fn: Type[Bar] = Bar) -> None:
        super(Baz, self).__init__()
        print(f'e => {e}')
        print(f'f => {f}')
        fn()


Wow = NewType('Wow', tp=Baz)


@app.register
def foo(x: str = 'wow', y_: Union[Type[Foo], Type[Wow]] = Wow, z_: Type[Wow] = Wow):
    print(f'x => {x}')
    print(f'y => {y_()}')
    print(f'z => {z_()}')


if __name__ == '__main__':
    app.run()
