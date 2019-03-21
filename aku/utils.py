import typing


def is_union(retype):
    return getattr(retype, '__origin__', None) is typing.Union


