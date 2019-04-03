from dataclasses import dataclass
from string import ascii_letters
from typing import List, Optional, Tuple, Type, TypeVar, Union

from hypothesis import assume, given, strategies as st

from aku.utils import is_union, unwrap_union
from tests.utils import Circle, Point

PRIMITIVES = (
    str, bool, int, float,
)


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
def union_type(draw):
    T = tuple(draw(st.lists(st.sampled_from([Point, Circle, Rect]), min_size=1, max_size=3, unique=True, )))
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
def optional(draw, strategy):
    R, T = draw(strategy)
    return Optional[R], T


@given(
    true=union(),
    false=st.one_of([
        primitives(),
        lists(),
        homo_tuple(),
        union_type(),
        type_var(),

        optional(primitives()),
        optional(lists()),
        optional(homo_tuple()),
        optional(union_type()),
        optional(type_var()),
    ])
)
def test_is_union(true, false):
    assert is_union(true[0])
    assert not is_union(false[0])


@given(
    true=union(),
)
def test_unwrap_union(true):
    assert unwrap_union(true[0]) == true[1]
