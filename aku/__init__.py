import sys
from typing import List, Tuple, Set, FrozenSet, Dict
from typing import Optional
from typing import Union, Type

if sys.version_info < (3, 8):
    from typing import _SpecialForm

    Literal = _SpecialForm('Literal', doc=
    """Special typing form to define literal types (a.k.a. value types).

    This form can be used to indicate to type checkers that the corresponding
    variable or function parameter has a value equivalent to the provided
    literal (or one of several literals):

      def validate_simple(data: Any) -> Literal[True]:  # always returns True
          ...

      MODE = Literal['r', 'rb', 'w', 'wb']
      def open_helper(file: str, mode: MODE) -> str:
          ...

      open_helper('/some/path', 'r')  # Passes type check
      open_helper('/other/path', 'typo')  # Error in type checker

   Literal[...] cannot be subclassed. At runtime, an arbitrary value
   is allowed as type argument to Literal[...], but type checkers may
   impose restrictions.
    """)
else:
    from typing import Literal

__all__ = [
    'List', 'Tuple', 'Set', 'FrozenSet', 'Dict',
    'Optional',
    'Union', 'Type',
    'Literal',
]
