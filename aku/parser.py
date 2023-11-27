from argparse import ArgumentParser as _ArgumentParser
from typing import Any
from typing import Tuple

from aku.formatter import AkuFormatter
from aku.utils import bool_type
from aku.utils import register_type


class ArgumentParser(_ArgumentParser):
    def __init__(self, prog: str = None,
                 usage: str = None,
                 description: str = None,
                 epilog: str = None,
                 parents: Tuple['_ArgumentParser', ...] = (),
                 formatter_class: Any = AkuFormatter,
                 prefix_chars: str = '-',
                 fromfile_prefix_chars: str = None,
                 argument_default: Any = None,
                 conflict_handler: str = 'error',
                 add_help: bool = False,
                 allow_abbrev: bool = False) -> None:
        super(ArgumentParser, self).__init__(
            prog=prog,
            usage=usage,
            description=description,
            epilog=epilog,
            parents=parents,
            formatter_class=formatter_class,
            prefix_chars=prefix_chars,
            fromfile_prefix_chars=fromfile_prefix_chars,
            argument_default=argument_default,
            conflict_handler=conflict_handler,
            add_help=add_help,
            allow_abbrev=allow_abbrev,
        )
        register_type(bool_type, self)

    def error(self, message: str) -> None:
        raise RuntimeError(message)
