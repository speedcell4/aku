from typing import List, Tuple, Set, FrozenSet

from aku import Aku

app = Aku()


@app.register
def foo(x: List[int]):
    print(f'x => {x}')


@app.register
def bar(x: Tuple[str, int, float]):
    print(f'x => {x}')


@app.register
def baz(x: Tuple[int, ...]):
    print(f'x => {x}')


@app.register
def qux(x: Set[int]):
    print(f'x => {x}')


@app.register
def quux(x: FrozenSet[int]):
    print(f'x => {x}')


if __name__ == '__main__':
    app.run()
