from aku.utils import fetch_name
from tests.utils import func, Class


def test_fetch_name():
    assert fetch_name(func) == 'func'
    assert fetch_name(Class) == 'class'
    assert fetch_name(Class()) == 'class'
    assert fetch_name(Class().method) == 'class.method'
    assert fetch_name(Class.class_method) == 'class.class_method'
    assert fetch_name(Class.static_method) == 'class.static_method'
