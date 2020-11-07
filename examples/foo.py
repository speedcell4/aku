from pathlib import Path

from aku import Aku

aku = Aku()


@aku.option
def foo(a: int, b: bool = True, c: str = '3', d: float = 4.0, e: Path = Path.home()):
    print(f'a => {a}')
    print(f'b => {b}')
    print(f'c => {c}')
    print(f'd => {d}')
    print(f'e => {e}')


aku.run()
