from typing import Type

from aku import Aku

app = Aku()


def foo(name: str = 'first'):
    print(f'{foo.__name__}.name => {name}')


def bar(name: str = 'second'):
    print(f'{bar.__name__}.name => {name}')


def baz(name: str = 'third'):
    print(f'{baz.__name__}.name => {name}')


@app.option
def reduplicated(fn1_: Type[foo] = foo, fn2_: Type[bar] = bar, fn3: Type[baz] = baz, **kwargs):
    print(kwargs['@aku'])
    fn1_()
    fn2_()
    fn3()


if __name__ == '__main__':
    app.run()
