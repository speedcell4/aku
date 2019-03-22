import typing
from argparse import SUPPRESS
from typing import Type, Union, Optional

from aku.utils import get_annotations, unwrap_function_union, is_function_union, is_union, is_optional
from tests.utils import Circle, Point


def test_is_is_union():
    assert is_union(typing.Union[str, int])
    assert is_union(typing.Union[Type[Point], Type[Circle]])

    assert not is_union(typing.List[str])
    assert not is_union(typing.Tuple[str])
    assert not is_union(typing.Union[str])
    assert not is_union(typing.Union[str, str])


def foo(a: int, b: int):
    raise NotImplementedError


def bar(a: int, b: int = 2):
    raise NotImplementedError


def baz(a: int = 1, b: int = 2):
    raise NotImplementedError


def qux(a: Type[Union[Circle, Point]]):
    raise NotImplementedError


def quux(a: (1, 2, 3)):
    raise NotImplementedError


def test_get_annotations_function():
    assert get_annotations(foo) == [('a', int, SUPPRESS, 'a'), ('b', int, SUPPRESS, 'b')]
    assert get_annotations(bar) == [('a', int, SUPPRESS, 'a'), ('b', int, 2, 'b')]
    assert get_annotations(baz) == [('a', int, 1, 'a'), ('b', int, 2, 'b')]
    assert get_annotations(qux) == [('a', Type[Union[Circle, Point]], SUPPRESS, 'a')]
    assert get_annotations(quux) == [('a', (1, 2, 3), SUPPRESS, 'a')]


def test_is_function_union():
    assert is_function_union(Type[Union[Point, Circle]])
    assert is_function_union(Type[Union[Point]])
    assert is_function_union(Type[Point])
    assert not is_function_union(Point)

    assert is_function_union(Optional[Type[Union[Point, Circle]]])
    assert is_function_union(Optional[Type[Union[Point]]])
    assert is_function_union(Optional[Type[Point]])
    assert not is_function_union(Optional[Point])


def test_unwrap_function_union():
    assert unwrap_function_union(Type[Union[Point, Circle]]) == (Circle, Point)
    assert unwrap_function_union(Type[Union[Point]]) == (Point,)
    assert unwrap_function_union(Type[Point]) == (Point,)
2