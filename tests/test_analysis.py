from string import ascii_letters

from hypothesis import given, strategies as st

from aku import Aku


def foo(a: int = 1, b: float = 2, c: complex = 3 + 4j, d: bool = True, e: str = 'e :: string'):
    return locals()


@given(
    a=st.integers(),
    b=st.floats(min_value=0),  # TODO check option
    # c=st.complex_numbers(min_magnitude=0),
    d=st.sampled_from([  # TODO generate case
        '0', 'f', 'false', 'n', 'no',
        '1', 't', 'true', 'y', 'yes',
    ]),
    e=st.text(min_size=1, alphabet=ascii_letters),
)
def test_foo(a, b, d, e):
    app = Aku()
    app.register(foo)
    ret = app.run([
        '--a', f'{a}',
        '--b', f'{b}',
        # '--c', f'{c}',
        '--d', f'{d}',
        '--e', f'{e}',
    ])
    assert ret['a'] == a
    assert ret['b'] == b
    if d in ('0', 'f', 'false', 'n', 'no'):
        assert not ret['d']
    else:
        assert ret['d']
    assert ret['e'] == e
