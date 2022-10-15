from typing import Type

from aku import Aku

app = Aku()


def foo(name: str = 'first'):
    print(f'{foo.__name__}.name => {name}')


def bar(name: str = 'second'):
    print(f'{bar.__name__}.name => {name}')


def baz(name: str = 'third'):
    print(f'{baz.__name__}.name => {name}')


@app.register
def main(a_: Type[foo] = foo, b_: Type[bar] = bar, c: Type[baz] = baz, **kwargs):
    a_()
    b_()
    c()
    print(kwargs['@aku'])


if __name__ == '__main__':
    app.run()
