from pathlib import Path
from typing import Literal

from aku import Aku

app = Aku()


@app.register
def foo(a: int = 1, b: str = '2', c: bool = True, d: float = 4.0, e: Path = Path.home()):
    print(f'a => {a}')
    print(f'b => {b}')
    print(f'c => {c}')
    print(f'd => {d}')
    print(f'e => {e}')


@app.register
def bar(name_with_underline: Literal['a', 'i', 'u', 'e', 'o'] = 'o'):
    print(f'name_with_underline => {name_with_underline}')


if __name__ == '__main__':
    app.run()
