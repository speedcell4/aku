from pathlib import Path
from typing import List, Tuple, Union, Type

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
def bar(x: Literal[1, 2, 3] = 2, y: List[int] = [2, 3, 4],
        z: Tuple[float, ...] = (), w: Tuple[float, str, int] = (1., '2', 3), **kwargs):
    print(f'{bar.__name__}.x => {x}')
    print(f'{bar.__name__}.y => {y}')
    print(f'{bar.__name__}.z => {z}')
    print(f'{bar.__name__}.w => {w}')
    if '@aku' in kwargs:
        print(f'{bar.__name__}.@aku => {kwargs["@aku"]}')


@aku.option
def delegate(call: Type[foo]):
    call()


@aku.option
def one(call: Union[Type[foo], Type[bar]]):
    call()


@aku.option
def both(a_: Type[foo], b_: Type[bar]):
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
