from string import ascii_letters

from hypothesis import given, strategies as st

from aku import Aku


@st.composite
def integers(draw):
    a = draw(st.integers())
    return f'{a}', a


@st.composite
def floats(draw):
    a = draw(st.floats(min_value=0))
    return f'{a}', a


@st.composite
def booleans(draw):
    a = draw(st.booleans())
    if a:
        r = draw(st.sampled_from(['1', 't', 'true', 'y', 'yes']))
    else:
        r = draw(st.sampled_from(['0', 'f', 'false', 'n', 'no']))
    return r, a


@st.composite
def strings(draw):
    a = draw(st.text(min_size=1, alphabet=ascii_letters))
    return a, a


@st.composite
def optional(draw, strategy):
    if draw(st.booleans()):
        return draw(strategy)
    r = draw(st.sampled_from(['nil', 'none', 'null']))
    return r, None


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
