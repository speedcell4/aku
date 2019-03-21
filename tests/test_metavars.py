from typing import Optional, Union

from aku.metavars import render_type


def test_render_parser():
    assert render_type(int) == 'Int'
    assert render_type(float) == 'Float'
    assert render_type(bool) == 'Bool'
    assert render_type(str) == 'Str'

    assert render_type(Optional[int]) == 'Int?'
    assert render_type(Union[int, str]) == '{Int, Str}'
    assert render_type(Optional[Union[int, str]]) == '{Int, Str}?'