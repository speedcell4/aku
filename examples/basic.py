from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from aku.tp import Tp
from typing import List, Tuple, Set, FrozenSet, Literal

parser = ArgumentParser(
    formatter_class=ArgumentDefaultsHelpFormatter,
)

Tp[str].add_argument(parser, 'pa', 'a')
Tp[int].add_argument(parser, 'pb', 1)
Tp[bool].add_argument(parser, 'pc', False)
Tp[float].add_argument(parser, 'pd', 2.0)

Tp[Literal['a', 'b', 'a']].add_argument(parser, 'la', 'a')
Tp[Literal[2, 3, 2]].add_argument(parser, 'lb', 3)
Tp[Literal[True, False, True]].add_argument(parser, 'lc', False)
Tp[Literal[2.0, 3.0]].add_argument(parser, 'ld', 2.0)

Tp[List[str]].add_argument(parser, 'sa', ['a', 'b'])
Tp[List[int]].add_argument(parser, 'sb', [2, 3])
Tp[List[bool]].add_argument(parser, 'sc', [True, False])
Tp[List[float]].add_argument(parser, 'sd', [2.0, 3.0])

Tp[Tuple[str, ...]].add_argument(parser, 'oa', ('a', 'b'))
Tp[Tuple[int, ...]].add_argument(parser, 'ob', (2, 3))
Tp[Tuple[bool, ...]].add_argument(parser, 'oc', (True, False))
Tp[Tuple[float, ...]].add_argument(parser, 'od', (2.0, 3.0))

Tp[Tuple[str, int]].add_argument(parser, 'ea', ('a', 1))
Tp[Tuple[int, int]].add_argument(parser, 'eb', (2, 1))
Tp[Tuple[bool, int]].add_argument(parser, 'ec', (True, 1))
Tp[Tuple[float, int]].add_argument(parser, 'ed', (2.0, 1))

Tp[Set[str]].add_argument(parser, 'ta', {'a'})
Tp[Set[int]].add_argument(parser, 'tb', {2})
Tp[Set[bool]].add_argument(parser, 'tc', {True})
Tp[Set[float]].add_argument(parser, 'td', {2.0})

# Tp[FrozenSet[str]].add_argument(parser, 'fa', frozenset({'a'}))
# Tp[FrozenSet[int]].add_argument(parser, 'fb', frozenset({2}))
# Tp[FrozenSet[bool]].add_argument(parser, 'fc', frozenset({True}))
# Tp[FrozenSet[float]].add_argument(parser, 'fd', frozenset({2.0}))

print(parser.parse_args())
