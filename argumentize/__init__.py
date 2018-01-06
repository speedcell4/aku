import argparse
import inspect
import itertools
import re

from .annotations import boolean

__all__ = [
    'App',
]

# TODO multiline docstring
PATTERN = re.compile(r'^:param (?P<param>\w+): (?P<doc>.*)$', re.MULTILINE)


class NoDefault(object):
    pass


class App(object):
    def __init__(self, prog: str = __name__, formatter_class=argparse.ArgumentDefaultsHelpFormatter) -> None:
        self._argument_parser = argparse.ArgumentParser(prog=prog, formatter_class=formatter_class)
        self._formatter_class = formatter_class
        self._functions = {}

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
        if name in self._functions:
            raise argparse.ArgumentTypeError(f'{name} was set already')
        self._functions[name] = func
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
            annotation = boolean if annotation is bool else annotation
            if isinstance(default, NoDefault):
                argument_parser.add_argument(
                    f'--{arg}', type=annotation, help=docs.get(arg, f'{arg}'), required=True,
                )
            else:
                argument_parser.add_argument(
                    f'--{arg}', type=annotation, help=docs.get(arg, f'{arg}'), default=annotation(default),
                )

    @property
    def subparsers(self):
        if not hasattr(self, '_subparsers'):
            setattr(self, '_subparsers',
                    self._argument_parser.add_subparsers(dest='subparser_name'))
        return getattr(self, '_subparsers')

    def run(self):
        for name, function in self._functions.items():
            if len(self._functions) == 1:
                parser = self._argument_parser
                self._argumentize(parser, function)
            else:
                self._argumentize(
                    self.subparsers.add_parser(name, formatter_class=self._formatter_class), function)
        args = vars(self._argument_parser.parse_args())
        if len(self._functions) == 1:
            return self._functions.popitem()[1](**args)
        return self._functions[args.pop('subparser_name')](**args)
