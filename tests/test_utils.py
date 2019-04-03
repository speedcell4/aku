from dataclasses import dataclass
from string import ascii_letters
from typing import List, Optional, Tuple, Type, TypeVar, Union

from hypothesis import assume, given, strategies as st

from aku.utils import is_homo_tuple, is_list, is_optional, is_type_union, is_type_var, is_union, is_value_union, \
    unwrap_homo_tuple, unwrap_list, unwrap_optional, unwrap_type_union, unwrap_type_var, unwrap_union, \
    unwrap_value_union

PRIMITIVES = (
    str, bool, int, float,
)


def add(x: int, y: int):
    return x + y


def sub(x: int, y: int):
    return x - y


def mul(x: int, y: int):
    return x * y


def div(x: int, y: int):
    return x / y


@dataclass
class Point(object):
    x: int
    y: int


@dataclass
class Circle(object):
    x: int
    y: int
    radius: int


@dataclass
class Rect(object):
    x: int
    y: int
    width: int
    height: int


CLASSES = (
    Point, Circle, Rect,
    add, sub, mul, div,
)


@st.composite
def _primitives(draw):
    return draw(st.sampled_from(PRIMITIVES))


@st.composite
def primitives(draw):
    T = draw(_primitives())
    return T, None


@st.composite
def union(draw, element=_primitives()):
    T = draw(st.lists(element, min_size=2, max_size=len(PRIMITIVES)).map(set))
    assume(len(T) > 1)
    return Union[tuple(T)], tuple(T)


@st.composite
def lists(draw, element=_primitives()):
    T = draw(element)
    return List[T], T


@st.composite
def homo_tuple(draw, element=_primitives()):
    T = draw(element)
    return Tuple[T, ...], T


@st.composite
def type_union(draw):
    T = tuple(draw(st.lists(st.sampled_from(CLASSES), min_size=1, max_size=3, unique=True, )))
    return Type[Union[T]], T


@st.composite
def type_var(draw):
    name = draw(st.text(min_size=1, max_size=5, alphabet=ascii_letters))
    T = tuple(draw(st.lists(
        st.sampled_from(CLASSES),
        min_size=2, max_size=3, unique=True,
    )))
    return TypeVar(name, *T), T


@st.composite
def value_union(draw):
    T = draw(_primitives())
    E = {
        str: st.text(),
        bool: st.booleans(),
        int: st.integers(),
        float: st.floats(),
    }[T]
    return tuple(draw(st.lists(E, min_size=1, max_size=10, unique=True))), T


@st.composite
def optional(draw, strategy):
    R, T = draw(strategy)
    return Optional[R], T


@given(
    true=union(),
    optional_true=optional(union()),
    false=st.one_of([
        primitives(),
        lists(),
        homo_tuple(),
        type_union(),
        type_var(),
        value_union(),

        optional(primitives()),
        optional(lists()),
        optional(homo_tuple()),
        optional(type_union()),
        optional(type_var()),
    ])
)
def test_is_union(true, optional_true, false):
    assert is_union(true[0])
    assert is_optional(optional_true[0])
    assert is_union(unwrap_optional(optional_true[0]))
    assert not is_union(false[0])


@given(
    true=union(),
    optional_true=optional(union()),
)
def test_unwrap_union(true, optional_true):
    assert unwrap_union(true[0]) == true[1]
    assert unwrap_union(unwrap_optional(optional_true[0])) == optional_true[1]


@given(
    true=lists(),
    optional_true=optional(lists()),
    false=st.one_of([
        primitives(),
        union(),
        homo_tuple(),
        type_union(),
        type_var(),
        value_union(),

        optional(primitives()),
        optional(union()),
        optional(homo_tuple()),
        optional(type_union()),
        optional(type_var()),
    ])
)
def test_is_lists(true, optional_true, false):
    assert is_list(true[0])
    assert is_optional(optional_true[0])
    assert is_list(unwrap_optional(optional_true[0]))
    assert not is_list(false[0])


@given(
    true=lists(),
    optional_true=optional(lists()),
)
def test_unwrap_lists(true, optional_true):
    assert unwrap_list(true[0]) == true[1]
    assert unwrap_list(unwrap_optional(optional_true[0])) == optional_true[1]


@given(
    true=homo_tuple(),
    optional_true=optional(homo_tuple()),
    false=st.one_of([
        primitives(),
        union(),
        lists(),
        type_union(),
        type_var(),
        value_union(),

        optional(primitives()),
        optional(union()),
        optional(lists()),
        optional(type_union()),
        optional(type_var()),
    ])
)
def test_is_homo_tuple(true, optional_true, false):
    assert is_homo_tuple(true[0])
    assert is_optional(optional_true[0])
    assert is_homo_tuple(unwrap_optional(optional_true[0]))
    assert not is_homo_tuple(false[0])


@given(
    true=homo_tuple(),
    optional_true=optional(homo_tuple()),
)
def test_unwrap_homo_tuple(true, optional_true):
    assert unwrap_homo_tuple(true[0]) == true[1]
    assert unwrap_homo_tuple(unwrap_optional(optional_true[0])) == optional_true[1]


@given(
    true=type_union(),
    optional_true=optional(type_union()),
    false=st.one_of([
        primitives(),
        union(),
        lists(),
        homo_tuple(),
        type_var(),
        value_union(),

        optional(primitives()),
        optional(union()),
        optional(lists()),
        optional(homo_tuple()),
        optional(type_var()),
    ])
)
def test_is_type_union(true, optional_true, false):
    assert is_type_union(true[0])
    assert is_optional(optional_true[0])
    assert is_type_union(unwrap_optional(optional_true[0]))
    assert not is_type_union(false[0])


@given(
    true=type_union(),
    optional_true=optional(type_union()),
)
def test_unwrap_type_union(true, optional_true):
    assert set(unwrap_type_union(true[0])) == set(true[1])
    assert set(unwrap_type_union(unwrap_optional(optional_true[0]))) == set(optional_true[1])


@given(
    true=type_var(),
    optional_true=optional(type_var()),
    false=st.one_of([
        primitives(),
        union(),
        lists(),
        homo_tuple(),
        type_union(),
        value_union(),

        optional(primitives()),
        optional(union()),
        optional(lists()),
        optional(homo_tuple()),
        optional(type_union()),
    ])
)
def test_is_type_var(true, optional_true, false):
    assert is_type_var(true[0])
    assert is_optional(optional_true[0])
    assert is_type_var(unwrap_optional(optional_true[0]))
    assert not is_type_var(false[0])


@given(
    true=type_var(),
    optional_true=optional(type_var()),
)
def test_unwrap_type_var(true, optional_true):
    assert unwrap_type_var(true[0]) == true[1]
    assert unwrap_type_var(unwrap_optional(optional_true[0])) == optional_true[1]


@given(
    true=value_union(),
    false=st.one_of([
        primitives(),
        union(),
        lists(),
        homo_tuple(),
        type_union(),
        type_var(),

        optional(primitives()),
        optional(union()),
        optional(lists()),
        optional(homo_tuple()),
        optional(type_union()),
        optional(type_var()),
    ])
)
def test_is_value_union(true, false):
    assert is_value_union(true[0])
    assert not is_value_union(false[0])


@given(
    true=value_union(),
)
def test_unwrap_value_union(true):
    assert unwrap_value_union(true[0]) == true[1]
