from typing import Union, Type

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


@app.register
def foo(x: str = 'wow', y: Union[Type[Foo], Type[Bar]] = Foo):
    print(f'x => {x}')
    print(f'y => {y()}')


if __name__ == '__main__':
    app.run()
