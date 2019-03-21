import typing
from typing import Type

from aku.utils import is_union, get_annotations
from tests.utils import Circle, Point


def test_is_is_union():
    assert is_union(typing.Union[str, int])
    assert is_union(typing.Union[Type[Point], Type[Circle]])

    assert not is_union(typing.List[str])
    assert not is_union(typing.Tuple[str])
    assert not is_union(typing.Union[str])
    assert not is_union(typing.Union[str, str])


def add0(a: int, b: int):
    raise NotImplementedError


def add1(a: int, b: int = 2):
    raise NotImplementedError


def add2(a: int = 1, b: int = 2):
    raise NotImplementedError


def test_get_annotations_function():
    assert get_annotations(add0) == [('a', int, '==SUPPRESS==', 'a'), ('b', int, '==SUPPRESS==', 'b')]
    assert get_annotations(add1) == [('a', int, '==SUPPRESS==', 'a'), ('b', int, 2, 'b')]
    assert get_annotations(add2) == [('a', int, 1, 'a'), ('b', int, 2, 'b')]
