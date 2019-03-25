from string import ascii_letters

from hypothesis import strategies as st


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
