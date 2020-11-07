from typing import Type, Union
from aku import Aku


def add(x: int, y: int):
    print(f'{x} + {y} => {x + y}')


def sub(x: int, y: int):
    print(f'{x} - {y} => {x - y}')


aku = Aku()


@aku.option
def one(op: Union[Type[add], Type[sub]]):
    op()


@aku.option
def both(lhs_: Type[add], rhs_: Type[sub]):
    lhs_()
    rhs_()


if __name__ == '__main__':
    aku.run()
