import argparse
import inspect
import itertools
import re

__all__ = [
    'App',
]

# TODO multiline docstring
PATTERN = re.compile(r'^:param (?P<param>\w+): (?P<doc>.*)$', re.MULTILINE)


class NoDefault(object):
    pass


def str2bool(string: str) -> bool:
    if string.lower() in ['y', 'yes', 't', 'true', '1']:
        return True
    if string.lower() in ['n', 'no', 'f', 'false', '0']:
        return False
    raise argparse.ArgumentTypeError(f'{string} can not be converted to boolean type')


class App(object):
    def __init__(self, prog: str = __name__, formatter_class=argparse.ArgumentDefaultsHelpFormatter) -> None:
        self._argument_parser = argparse.ArgumentParser(prog=prog, formatter_class=formatter_class)
        self.funcs = {}

    @staticmethod
    def _arg_default_annotation_stream(args, defaults, annotations):
        for index, arg in enumerate(args[::-1], start=1):
            if index <= len(defaults):
                yield arg, annotations.get(arg, str), defaults[-index]
            else:
                yield arg, annotations.get(arg, str), NoDefault()

    @staticmethod
    def _kwonlyarg_default_annotation_stream(kwonlyargs, kwonlydefaults, annotations):
        for arg in kwonlyargs:
            yield arg, annotations.get(arg, str), kwonlydefaults.get(arg, NoDefault())

    @staticmethod
    def _argdocs(func):
        docstring = inspect.getdoc(func) or ''
        return {key: value for key, value in re.findall(PATTERN, docstring)}

    def register(self, func):
        name = func.__name__
        if name in self.funcs:
            raise argparse.ArgumentError(f'{name} was set already')
        self.funcs[name] = func
        return func

    def _argumentize(self, argument_parser, func):
        argspec = inspect.getfullargspec(func)

        args = argspec.args
        defaults = argspec.defaults or []
        kwonlyargs = argspec.kwonlyargs
        kwonlydefaults = argspec.kwonlydefaults or {}
        annotations = argspec.annotations
        docs = self._argdocs(func)

        stream = itertools.chain(
            reversed(list(self._arg_default_annotation_stream(args, defaults, annotations))),
            self._kwonlyarg_default_annotation_stream(kwonlyargs, kwonlydefaults, annotations),
        )

        for arg, annotation, default in stream:
            annotation = str2bool if annotation is bool else annotation
            if isinstance(default, NoDefault):
                argument_parser.add_argument(
                    f'--{arg}', type=annotation, help=docs.get(arg, f'{arg}'), required=True,
                )
            else:
                argument_parser.add_argument(
                    f'--{arg}', type=annotation, help=docs.get(arg, f'{arg}'), default=default,
                )

    def run(self):
        subparsers = self._argument_parser.add_subparsers(dest='subparser_name')
        for name, func in self.funcs.items():
            if len(self.funcs) == 1:
                parser = self._argument_parser
                self._argumentize(parser, func)
            else:
                self._argumentize(subparsers.add_parser(name), func)
        args = vars(self._argument_parser.parse_args())
        return self.funcs[args.pop('subparser_name')](**args)
