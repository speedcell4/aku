from typing import List, Tuple, Set, FrozenSet

from aku import Aku

app = Aku()


@app.option
def foo(x: List[int]):
    print(f'x => {x}')


@app.option
def bar(x: Tuple[str, int, float]):
    print(f'x => {x}')


@app.option
def baz(x: Tuple[int, ...]):
    print(f'x => {x}')


@app.option
def qux(x: Set[int]):
    print(f'x => {x}')


@app.option
def quux(x: FrozenSet[int]):
    print(f'x => {x}')


if __name__ == '__main__':
    app.run()
