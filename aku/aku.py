import functools
import inspect
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, Namespace, SUPPRESS
from typing import Type, Union

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
        self._functions.append(Type[fn])
        return fn

    def parse_args(self, args=None) -> Namespace:
        AkuTp[Union[tuple(self._functions)]].add_argument(
            argument_parser=self, name='root', default=SUPPRESS,
            prefixes=(), domain=(),
        )

        namespace, args = None, None
        while True:
            namespace, args = self.parse_known_args(args=args, namespace=namespace)
            if hasattr(self, NEW_ACTIONS):
                self._actions = self._actions + getattr(self, NEW_ACTIONS)
                delattr(self, NEW_ACTIONS)
            else:
                break

        return namespace

    def run(self, namespace: Namespace = None):
        if namespace is None:
            namespace = self.parse_args()
        if isinstance(namespace, Namespace):
            namespace = namespace.__dict__

        args = {}
        for key, value in namespace.items():
            collection = args
            *names, key = key.split('.')
            for name in names:
                collection = collection.setdefault(name, {})
            if key == '@fn':
                collection[key] = value
            else:
                collection.setdefault('@args', {})[key] = value

        def recur(x):
            if isinstance(x, dict):
                if '@fn' in x:
                    kwargs = {key: recur(value) for key, value in x['@args'].items()}
                    return functools.partial(x['@fn'], **kwargs)
                else:
                    return {
                        key: recur(value)
                        for key, value in x.items()
                    }
            else:
                return x

        ret = recur(args)
        assert len(ret) == 1
        for _, fn in ret.items():
            if inspect.getfullargspec(fn).varkw is None:
                return fn()
            else:
                return fn(**{'@aku': args})
