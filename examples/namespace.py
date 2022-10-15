from typing import Type

from aku import Aku

app = Aku()


def foo(x: str = 'meow'):
    print(f'x => {x}')


def bar(a_: Type[foo] = foo):
    a_()


def qux(b: Type[bar] = bar):
    b()


@app.register
def quux(c_: Type[qux] = qux, **kwargs):
    c_()
    print(kwargs['@aku'])


if __name__ == '__main__':
    app.run()
