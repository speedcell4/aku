import sys
from typing import List, Tuple, Set, FrozenSet
from typing import Optional
from typing import Type
from typing import Union

if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing import _SpecialForm

    Literal = _SpecialForm('Literal', doc='Literal')

from .aku import Aku

__version__ = '0.2.1'

__all__ = [
    'Type',
    'List', 'Tuple', 'Set', 'FrozenSet',
    'Union', 'Optional', 'Literal',

    'Aku',
]
