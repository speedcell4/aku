from argparse import ArgumentParser as _ArgumentParser

from aku.formatter import AkuFormatter
from aku.utils import register_type, bool_type


class ArgumentParser(_ArgumentParser):
    def __init__(self, prog=None,
                 usage=None,
                 description=None,
                 epilog=None,
                 parents=(),
                 formatter_class=AkuFormatter,
                 prefix_chars='-',
                 fromfile_prefix_chars=None,
                 argument_default=None,
                 conflict_handler='error',
                 add_help=True,
                 allow_abbrev=False) -> None:
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
