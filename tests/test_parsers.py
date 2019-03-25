import argparse

import pytest

from aku.parsers import get_parsing_fn
from aku.utils import NoneType


def test_str2null():
    func = get_parsing_fn(NoneType)

    assert func('nil') is None
    assert func('null') is None
    assert func('none') is None

    with pytest.raises(argparse.ArgumentTypeError):
        assert func('nil2') is None
    with pytest.raises(argparse.ArgumentTypeError):
        assert func('null3') is None
    with pytest.raises(argparse.ArgumentTypeError):
        assert func('none4') is None


def test_str2bool():
    func = get_parsing_fn(bool)

    assert func('0') is False
    assert func('F') is False
    assert func('FaLSe') is False
    assert func('nO') is False
    assert func('n') is False

    assert func('1') is True
    assert func('t') is True
    assert func('True') is True
    assert func('yeS') is True
    assert func('y') is True

    with pytest.raises(argparse.ArgumentTypeError):
        assert func('yess') is None
    with pytest.raises(argparse.ArgumentTypeError):
        assert func('n0') is None
