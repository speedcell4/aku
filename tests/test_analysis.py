import inspect
from typing import List, Tuple, Union, Type

from hypothesis import given, strategies as st

from aku import Aku
from tests.strategies import booleans, floats, integers, optional, strings


def foo(a: int = 1, b: float = 2, c: complex = 3 + 4j, d: bool = True, e: str = 'e :: string'):
    return locals()


@given(
    a=integers(),
    b=floats(),  # TODO check option
    # c=st.complex_numbers(min_magnitude=0),
    d=booleans(),
    e=strings(),
)
def test_foo(a, b, d, e):
    app = Aku()
    app.register(foo)
    ret = app.run([
        '--a', f'{a[0]}',
        '--b', f'{b[0]}',
        # '--c', f'{c[0]}',
        '--d', f'{d[0]}',
        '--e', f'{e[0]}',
    ])
    assert ret['a'] == a[1]
    assert ret['b'] == b[1]
    assert ret['d'] == d[1]
    assert ret['e'] == e[1]


# TODO there does not exist optional[str]?
def bar(a: int = None, b: float = None, c: complex = None, d: bool = None, e: str = None):
    return locals()


@given(
    a=optional(integers()),
    b=optional(floats()),  # TODO check option
    # c=st.complex_numbers(min_magnitude=0),
    d=optional(booleans()),
    e=strings(),
)
def test_bar(a, b, d, e):
    app = Aku()
    app.register(bar)
    ret = app.run([
        '--a', f'{a[0]}',
        '--b', f'{b[0]}',
        # '--c', f'{c[0]}',
        '--d', f'{d[0]}',
        '--e', f'{e[0]}',
    ])
    assert ret['a'] == a[1]
    assert ret['b'] == b[1]
    assert ret['d'] == d[1]
    assert ret['e'] == e[1]


def baz(a: Union[int, str] = 2, b: Union[float, float] = 3.0, c: Union[int, float] = None):
    return locals()


@given(
    a=st.one_of(integers(), strings()),
    b=floats(),
    c=optional(st.one_of(integers(), floats())),
)
def test_baz(a, b, c):
    app = Aku()
    app.register(baz)
    ret = app.run([
        '--a', f'{a[0]}',
        '--b', f'{b[0]}',
        '--c', f'{c[0]}',
    ])
    assert ret['a'] == a[1]
    assert ret['b'] == b[1]
    assert ret['c'] == c[1]


def qux(a: List[int] = [6, 7], b: Tuple[float, ...] = (8.0, 9.0)):
    return locals()


@given(
    a=st.lists(integers()),
    b=st.lists(floats()),
)
def test_qux(a, b):
    if len(a) == 0:
        a_reprs, a_values = [], inspect.getfullargspec(qux).defaults[0]
    else:
        a_reprs, a_values = map(list, zip(*a))

    if len(b) == 0:
        b_reprs, b_values = (), inspect.getfullargspec(qux).defaults[1]
    else:
        b_reprs, b_values = map(tuple, zip(*b))

    app = Aku()
    app.register(qux)

    args = []
    for a_repr in a_reprs:
        args.extend(['--a', f'{a_repr}'])
    for b_repr in b_reprs:
        args.extend(['--b', f'{b_repr}'])
    ret = app.run(args)

    assert ret['a'] == a_values
    assert ret['b'] == b_values


def add(x: int, y: int):
    return x + y


def sub(x: int, y: int):
    return x - y


def corge(cal: Type[Union[add, sub]] = add):
    return cal()


@given(
    x=integers(),
    y=integers(),
)
def test_corge_add(x, y):
    app = Aku()
    app.register(corge)

    ret = app.run([
        '--cal', 'add',
        '--x', f'{x[0]}',
        '--y', f'{y[0]}',
    ])

    assert ret == x[1] + y[1]


@given(
    x=integers(),
    y=integers(),
)
def test_corge_sub(x, y):
    app = Aku()
    app.register(corge)

    ret = app.run([
        '--cal', 'sub',
        '--x', f'{x[0]}',
        '--y', f'{y[0]}',
    ])

    assert ret == x[1] - y[1]
