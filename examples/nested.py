from typing import Type
from typing import Union

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


@app.register
def foo(x: str = 'wow', y_: Union[Type[Foo], Type[Baz]] = Baz):
    print(f'x => {x}')
    print(f'y => {y_()}')


@app.register
def bar(x: str = 'wow', y_: Union[Type[Foo], Type[Baz]] = Baz):
    print(f'x => {x}')
    print(f'y => {y_()}')


if __name__ == '__main__':
    app.run()
