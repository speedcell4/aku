from aku import Aku

aku = Aku()


@aku.register
def add(a: int, b: int = 2):
    print(f'{a} + {b} => {a + b}')


@aku.register
def say_hello(name: str):
    print(f'hello {name}')


aku.run()
