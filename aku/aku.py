import functools
import inspect
import sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, Namespace, SUPPRESS
from typing import Type

from aku.tp import AkuTp
from aku.utils import _init_argument_parser, NEW_ACTIONS, fetch_name, AKU, AKU_FN, AKU_ROOT


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
                 allow_abbrev=False) -> None:
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

    def parse_args(self, args=None) -> Namespace:
        assert len(self._functions) > 0

        namespace, args, argument_parser = None, sys.argv, self
        if len(self._functions) == 1:
            fn = self._functions[0]
            AkuTp[Type[fn]].add_argument(
                argument_parser=argument_parser, name=AKU_ROOT, default=SUPPRESS,
                prefixes=(), domain=(),
            )
        else:
            subparsers = self.add_subparsers()
            functions = {}
            for fn in self._functions:
                name = fetch_name(fn)
                if name not in functions:
                    functions[name] = (fn, subparsers.add_parser(name=name))
                else:
                    raise ValueError(f'{name} was already registered')

            if len(args) > 1 and args[1] in functions:
                fn, argument_parser = functions[args[1]]
                AkuTp[Type[fn]].add_argument(
                    argument_parser=argument_parser, name=AKU_ROOT, default=SUPPRESS,
                    prefixes=(), domain=(),
                )

        while True:
            namespace, args = argument_parser.parse_known_args(args=args, namespace=namespace)
            if hasattr(argument_parser, NEW_ACTIONS):
                argument_parser._actions = argument_parser._actions + getattr(argument_parser, NEW_ACTIONS)
                delattr(argument_parser, NEW_ACTIONS)
            else:
                break

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
            if key == AKU_FN:
                curry_co[key], literal_co[key] = value
            else:
                curry_co[key] = literal_co[key] = value

        def recur(item):
            if isinstance(item, dict):
                if AKU_FN in item:
                    func = item.pop(AKU_FN)
                    kwargs = {k: recur(v) for k, v in item.items()}
                    return functools.partial(func, **kwargs)
                else:
                    return {k: recur(v) for k, v in item.items()}
            else:
                return item

        ret = recur(curry)
        assert len(ret) == 1
        for _, fn in ret.items():
            if inspect.getfullargspec(fn).varkw is None:
                return fn()
            else:
                return fn(**{AKU: literal})
