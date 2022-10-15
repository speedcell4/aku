from pathlib import Path

from aku import Aku, Literal

app = Aku()


@app.register
def foo(a: int = 1, b: str = '2', c: bool = True, d: float = 4.0, e: Path = Path.home()):
    print(f'a => {a}')
    print(f'b => {b}')
    print(f'c => {c}')
    print(f'd => {d}')
    print(f'e => {e}')


@app.register
def bar(a: Literal['a', 'i', 'u', 'e', 'o'] = 'o'):
    print(f'a => {a}')


if __name__ == '__main__':
    app.run()
