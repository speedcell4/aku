from aku import Aku

aku = Aku()


@aku.register
def add(a: int, b: int = 2):
    print(f'{a} + {b} => {a + b}')


aku.run()
