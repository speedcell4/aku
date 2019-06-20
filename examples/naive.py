from typing import TypeVar

import aku

app = aku.Aku()


@app.register
def add(x: int = 1, y: int = 2):
    return x + y


@app.register
def sub(x: int = 3, y: int = 4):
    return x - y


@app.register
def mul(encoder: TypeVar('enc', add, sub) = sub):
    print(f'op() => {encoder()}')


if __name__ == '__main__':
    app.run()
