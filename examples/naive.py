from typing import TypeVar, Callable

import aku

app = aku.Aku()


@app.register
def add(x: int = 1, y: int = 2):
    return x + y


@app.register
def sub(a: int = 3, b: int = 4):
    return a - b


@app.register
def mul(encoder: TypeVar('enc', add, sub) = sub):
    print(f'op() => {encoder()}')


@app.register
def nice(a: Callable = add, b: Callable = sub):
    ret = a() + b()
    print(f'ret => {ret}')


if __name__ == '__main__':
    app.run()
