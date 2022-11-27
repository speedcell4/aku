from aku import Aku

aku = Aku()


@aku.register
def foo(a: int = 1, /, b: int = 2, c: int = 3, *, d: int = 4, **kwargs):
    print(f'a => {a}')
    print(f'b => {b}')
    print(f'c => {c}')
    print(f'd => {d}')
    print(f'kwargs => {kwargs}')


if __name__ == '__main__':
    aku.run()
