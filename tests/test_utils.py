import typing
from typing import Type

from aku.utils import is_union
from tests.utils import Circle, Point


def test_is_is_union():
    assert is_union(typing.Union[str, int])
    assert is_union(typing.Union[Type[Point], Type[Circle]])

    assert not is_union(typing.List[str])
    assert not is_union(typing.Tuple[str])
    assert not is_union(typing.Union[str])
    assert not is_union(typing.Union[str, str])
