from typing import Type

from aku import Aku

app = Aku()


def foo(x: str):
    print(f'x => {x}')


def bar(fn_: Type[foo] = foo):
    fn_()


def qux(fn: Type[bar] = bar):
    fn()


@app.register
def quux(fn_: Type[qux] = qux, **kwargs):
    fn_()
    print(kwargs['@aku'])


if __name__ == '__main__':
    app.run()
