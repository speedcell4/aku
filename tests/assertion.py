def assert_equal(actual, excepted):
    if isinstance(excepted, list):
        assert isinstance(actual, list)
        for a, e in zip(actual, excepted):
            assert_equal(a, e)

    elif isinstance(excepted, tuple):
        assert isinstance(actual, tuple)
        for a, e in zip(actual, excepted):
            assert_equal(a, e)

    elif isinstance(excepted, (set, frozenset)):
        assert isinstance(actual, (set, frozenset))
        assert frozenset(actual) == frozenset(excepted), f'{frozenset(actual)} != {frozenset(excepted)}'

    elif isinstance(excepted, dict):
        assert isinstance(actual, dict)
        assert_equal(actual=frozenset(actual.keys()), excepted=frozenset(excepted.keys()))

        for key in excepted.keys():
            assert_equal(actual=actual[key], excepted=excepted[key])

    else:
        assert actual == excepted, f'{actual} != {excepted}'
