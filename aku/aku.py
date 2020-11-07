import functools
import inspect
import sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, Namespace, SUPPRESS
from typing import Type

from aku.tp import AkuTp
from aku.utils import _init_argument_parser, NEW_ACTIONS


class Aku(ArgumentParser):
    def __init__(self, prog=None,
                 usage=None,
                 description=None,
                 epilog=None,
                 parents=(),
                 formatter_class=functools.partial(
                     ArgumentDefaultsHelpFormatter,
                     max_help_position=82,
                 ),
                 prefix_chars='-',
                 fromfile_prefix_chars=None,
                 argument_default=None,
                 conflict_handler='error',
                 add_help=True,
                 allow_abbrev=True) -> None:
        super(Aku, self).__init__(
            prog=prog, usage=usage, description=description, epilog=epilog,
            parents=parents, formatter_class=formatter_class, prefix_chars=prefix_chars,
            fromfile_prefix_chars=fromfile_prefix_chars, argument_default=argument_default,
            conflict_handler=conflict_handler, add_help=add_help, allow_abbrev=allow_abbrev,
        )
        _init_argument_parser(self)

        self._functions = []

    def option(self, fn):
        self._functions.append(fn)
        return fn

    @staticmethod
    def _add_root_function(argument_parser, fn, name):
        AkuTp[Type[fn]].add_argument(
            argument_parser=argument_parser, name='@root', default=SUPPRESS,
            prefixes=(), domain=(),
        )
        return Namespace(**{'@root.@fn': (fn, name)})

    def parse_args(self, args=None) -> Namespace:
        assert len(self._functions) > 0

        namespace, args, argument_parser = None, sys.argv, self
        if len(self._functions) == 1:
            fn = self._functions[0]
            namespace = self._add_root_function(argument_parser, fn, fn.__name__)
        else:
            subparsers = self.add_subparsers()
            functions = {
                fn.__name__: (fn, subparsers.add_parser(name=fn.__name__))
                for fn in self._functions
            }
            if len(args) > 1 and args[1] in functions:
                fn, argument_parser = functions[args[1]]
                namespace = self._add_root_function(argument_parser, fn, args[1])

        while True:
            namespace, args = argument_parser.parse_known_args(args=args, namespace=namespace)
            print(f'namespace => {namespace}')
            print(f'args => {args}')
            if hasattr(argument_parser, NEW_ACTIONS):
                argument_parser._actions = argument_parser._actions + getattr(argument_parser, NEW_ACTIONS)
                delattr(argument_parser, NEW_ACTIONS)
            else:
                break

        print(f'namespace => {namespace}')
        return namespace

    def run(self, namespace: Namespace = None):
        if namespace is None:
            namespace = self.parse_args()
        if isinstance(namespace, Namespace):
            namespace = namespace.__dict__

        curry, literal = {}, {}
        for key, value in namespace.items():
            curry_co = curry
            literal_co = literal
            *names, key = key.split('.')
            for name in names:
                curry_co = curry_co.setdefault(name, {})
                literal_co = literal_co.setdefault(name, {})
            if key == '@fn':
                curry_co[key] = value[0]
                literal_co[key] = value[1]
            else:
                curry_co[key] = literal_co[key] = value

        print(f'curry => {curry}')
        print(f'literal => {literal}')

        def recur(x):
            if isinstance(x, dict):
                if '@fn' in x:
                    fn = x.pop('@fn')
                    kwargs = {key: recur(value) for key, value in x.items()}
                    return functools.partial(fn, **kwargs)
                else:
                    return {
                        key: recur(value)
                        for key, value in x.items()
                    }
            else:
                return x

        print(curry)
        ret = recur(curry)
        print(f'ret => {ret}')
        assert len(ret) == 1
        for _, fn in ret.items():
            if inspect.getfullargspec(fn).varkw is None:
                return fn()
            else:
                return fn(**{'@aku': curry})
