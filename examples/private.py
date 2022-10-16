from typing import Type

from aku import Aku

app = Aku()


def foo(a: int = 1, b: int = 2):
    print(f'foo.a => {a}')
    print(f'foo.b => {b}')


def bar(c: int = 1, d: int = 2):
    print(f'bar.c => {c}')
    print(f'bar.d => {d}')


@app.register
def main(_: int = 1, _x: Type[foo] = foo, __y: Type[bar] = bar, **kwargs):
    print(f'_ => {_}')
    _x()
    __y()
    print(kwargs['@aku'])


if __name__ == '__main__':
    app.run()
