from pathlib import Path
from typing import Union

from aku import Aku, Literal

aku = Aku()


@aku.option
def foo(x: int, y: str = '4', z: bool = True, w: Path = Path.home(), **kwargs):
    print(f'{foo.__name__}.x => {x}')
    print(f'{foo.__name__}.y => {y}')
    print(f'{foo.__name__}.z => {z}')
    print(f'{foo.__name__}.w => {w}')
    if '@aku' in kwargs:
        print(f'{foo.__name__}.@aku => {kwargs["@aku"]}')


@aku.option
def bar(x: Literal[1, 2, 3] = 2, y: list[int] = [2, 3, 4],
        z: tuple[float, ...] = (), w: tuple[float, str, int] = (1., '2', 3), **kwargs):
    print(f'{bar.__name__}.x => {x}')
    print(f'{bar.__name__}.y => {y}')
    print(f'{bar.__name__}.z => {z}')
    print(f'{bar.__name__}.w => {w}')
    if '@aku' in kwargs:
        print(f'{bar.__name__}.@aku => {kwargs["@aku"]}')


@aku.option
def delegate(call: type[foo]):
    call()


@aku.option
def one(call: Union[type[foo], type[bar]]):
    call()


@aku.option
def both(a_: type[foo], b_: type[bar]):
    a_()
    b_()


class A(object):
    @classmethod
    def baz(cls, x: int):
        print(f'{A.__name__}.x => {x}')


aku.option(A.baz)


class B(object):
    @classmethod
    def baz(cls, x: int):
        print(f'{B.__name__}.x => {x}')


aku.option(B.baz)

print(aku.run())
