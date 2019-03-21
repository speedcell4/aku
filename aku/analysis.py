import argparse
import sys
from typing import Callable

from aku.arguments import expand_function


class Aku(object):
    def __init__(self) -> None:
        super(Aku, self).__init__()
        self.parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        self.functions = {}

    def register(self, func: Callable, name: str = None) -> Callable:
        if name is None:
            name = func.__name__
        assert name not in self.functions

        self.functions[name] = func
        return func

    def run(self, args=None, namespace=None):
        if self.functions.__len__() == 1:
            func = list(self.functions.values())[0]
            expand_function(func, self.parser)
        else:
            subparsers = self.parser.add_subparsers()
            parsers = {
                name: subparsers.add_parser(name)
                for name, _ in self.functions.items()
            }
            if sys.argv.__len__() > 1 and sys.argv[1] in parsers:
                func = self.functions[sys.argv[1]]
                expand_function(func, parsers[sys.argv[1]])

        kwargs = self.parser.parse_args(args=args, namespace=namespace)
        return func(**kwargs.__dict__)
