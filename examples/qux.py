from typing import Type

from aku import Aku


def add(x: int, y: int):
    print(f'{x} + {y} => {x + y}')


def sub(x: int, y: int):
    print(f'{x} - {y} => {x - y}')


aku = Aku()


@aku.option
def both(a_: Type[add], b_: Type[sub]):
    a_()
    b_()


@aku.option
def work(nice_: Type[both]):
    nice_()


if __name__ == '__main__':
    aku.run()
