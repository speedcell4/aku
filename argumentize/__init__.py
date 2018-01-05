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
    raise argparse.ArgumentError(string, 'can not be converted to boolean value')


class App(object):
    def __init__(self, prog: str = __file__, formatter_class=argparse.ArgumentDefaultsHelpFormatter) -> None:
        self.prog = prog
        self.formatter_class = formatter_class
        self.argument_parser = argparse.ArgumentParser(
            prog=self.prog, formatter_class=formatter_class)

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
    def argdocs(func):
        docstring = inspect.getdoc(func) or ''
        return {key: value for key, value in re.findall(PATTERN, docstring)}

    def argumentize(self, func):
        # TODO multi functions
        self.func = func
        argspec = inspect.getfullargspec(func)

        args = argspec.args
        defaults = argspec.defaults or []
        kwonlyargs = argspec.kwonlyargs
        kwonlydefaults = argspec.kwonlydefaults or {}
        annotations = argspec.annotations
        docs = self.argdocs(func)

        stream = itertools.chain(
            reversed(list(self._arg_default_annotation_stream(args, defaults, annotations))),
            self._kwonlyarg_default_annotation_stream(kwonlyargs, kwonlydefaults, annotations),
        )

        for arg, annotation, default in stream:
            annotation = str2bool if annotation is bool else annotation
            if isinstance(default, NoDefault):
                self.argument_parser.add_argument(
                    f'--{arg}', required=True, type=annotation, help=docs.get(arg, f'{arg}')
                )
            else:
                self.argument_parser.add_argument(
                    f'--{arg}', type=annotation, help=docs.get(arg, f'{arg}'), default=default,
                )

        return func

    def run(self):
        args = self.argument_parser.parse_args()
        return self.func(**vars(args))
