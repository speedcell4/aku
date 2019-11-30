import sys
from typing import FrozenSet
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple
from typing import Type
from typing import Union

if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing import _SpecialForm

    Literal = _SpecialForm('Literal', doc='Literal')

from aku.aku import Aku

__version__ = '0.2.1'

__all__ = [
    'Type', 'Union', 'Optional', 'Literal',
    'List', 'Tuple', 'Set', 'FrozenSet',
    'Aku',
]
