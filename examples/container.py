from typing import FrozenSet
from typing import List
from typing import Set
from typing import Tuple

from aku import Aku

app = Aku()


@app.register
def foo(x: List[int] = [1, 2],
        y: Tuple[str, int] = ('3', 4),
        z: Tuple[int, ...] = (5, 6, 7),
        a: Set[int] = {8, 9},
        b: FrozenSet[int] = frozenset([10, 11, 12])):
    print(f'x => {x}')
    print(f'y => {y}')
    print(f'z => {z}')
    print(f'a => {a}')
    print(f'b => {b}')


if __name__ == '__main__':
    app.run()
