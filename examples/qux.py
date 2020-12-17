from typing import Type, Union

from aku import Aku


def add(x: int, y: int):
    print(f'{x} + {y} => {x + y}')


def sub(x: int, y: int):
    print(f'{x} - {y} => {x - y}')


aku = Aku()


@aku.option
def op(a_: Type[add], b_: Type[sub], c_: Union[Type[add], Type[sub]]):
    a_()
    b_()
    c_()


@aku.option
def nested(fn_: Union[Type[op], Type[add]]):
    fn_()


if __name__ == '__main__':
    aku.run()
