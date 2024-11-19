from aku.utils import get_name
from tests.utils import Class, func


def test_fetch_name():
    assert get_name(func).lower() == 'func'
    assert get_name(Class).lower() == 'class'
    assert get_name(Class()).lower() == 'class'
    assert get_name(Class().method).lower() == 'class.method'
    assert get_name(Class.class_method).lower() == 'class.class_method'
    assert get_name(Class.static_method).lower() == 'class.static_method'
