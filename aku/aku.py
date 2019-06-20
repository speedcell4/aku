import sys
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from functools import partial

from aku import add_option


class Aku(object):
    def __init__(self, prog: str = None, usage: str = None, description: str = None):
        super(Aku, self).__init__()
        self.parser = ArgumentParser(
            prog=prog, usage=usage, description=description,
            formatter_class=ArgumentDefaultsHelpFormatter,
        )
        self._funcs = {}

    def register(self, func):
        self._funcs[func.__name__] = func
        return func

    def run(self, args=None, namespace=None):
        self.delays = []

        if len(self._funcs) == 1:
            fn = list(self._funcs.values())[0]
            add_option.add_function(
                parser=self.parser, prefix=None, name=None,
                annotation=fn, default=None, delays=self.delays,
            )
        else:
            subparsers = self.parser.add_subparsers()
            parsers = {
                name: subparsers.add_parser(name)
                for name, _ in self._funcs.items()
            }
            if sys.argv.__len__() > 1 and sys.argv[1] in parsers:
                fn = self._funcs[sys.argv[1]]
                add_option.add_function(
                    parser=parsers[sys.argv[1]], prefix=None, name=None,
                    annotation=fn, default=None, delays=self.delays,
                )

        args, _ = self.parser.parse_known_args(args, namespace)
        self.raw_args = {k: v for k, v in vars(args).items()}
        self.args = {k: v for k, v in vars(args).items()}

        for dest, name, key in reversed(self.delays):
            self.args[dest] = partial(self.args[dest], **{name: self.args[key]})
            del self.args[key]

            obj_dest = f'@{dest}'
            if obj_dest in self.args:
                del self.args[obj_dest]
            if dest in self.raw_args and obj_dest in self.raw_args:
                self.raw_args[dest] = self.raw_args[obj_dest]
                del self.raw_args[obj_dest]

        return fn(**self.args)
