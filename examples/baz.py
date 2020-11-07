from typing import List, Tuple

from aku import Aku, Literal

aku = Aku()


@aku.option
def baz(a: List[int], b: Tuple[bool, ...], c: Tuple[int, bool, str], d: Literal[42, 1905]):
    print(f'a => {a}')
    print(f'b => {b}')
    print(f'c => {c}')
    print(f'd => {d}')


if __name__ == '__main__':
    aku.run()
