import functools
import inspect
import sys
from argparse import ArgumentParser, Namespace, SUPPRESS
from typing import Type, List

from aku.formatter import AkuFormatter
from aku.tp import AkuTp
from aku.utils import init_argument_parser, get_name, AKU_FN, AKU, AKU_DELAY


class Aku(ArgumentParser):
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
                 always_use_subparse=False,
                 allow_abbrev=False) -> None:
        super(Aku, self).__init__(
            prog=prog, usage=usage, description=description, epilog=epilog,
            parents=parents, formatter_class=formatter_class, prefix_chars=prefix_chars,
            fromfile_prefix_chars=fromfile_prefix_chars, argument_default=argument_default,
            conflict_handler=conflict_handler, add_help=False, allow_abbrev=allow_abbrev,
        )
        init_argument_parser(self)

        self.options = []
        self.add_help = add_help
        self.always_use_subparse = always_use_subparse

    def option(self, fn):
        self.options.append(fn)
        return fn

    def _parse(self, args: List[str] = None, namespace: Namespace = None):
        assert len(self.options) > 0, f'you are supposed to .option at least one callable'

        if args is None:
            args = sys.argv[1:]

        argument_parser = self
        if not self.always_use_subparse and len(self.options) == 1:
            option = self.options[0]
            AkuTp[Type[option]].add_argument(
                argument_parser=argument_parser,
                name=AKU, default=SUPPRESS,
                prefixes=(), domain=(),
            )
        else:
            subparsers = argument_parser.add_subparsers()
            options = {}
            for option in self.options:
                name = get_name(option)
                if name not in options:
                    options[name] = (option, subparsers.add_parser(name=name))
                else:
                    raise ValueError(f'{name} was already registered')

            if len(args) > 0 and args[0] in options:
                arg, *args = args
                option, argument_parser = options[arg]
                AkuTp[Type[option]].add_argument(
                    argument_parser=argument_parser,
                    name=AKU, default=SUPPRESS,
                    prefixes=(), domain=(),
                )

        while True:
            namespace, args = argument_parser.parse_known_args(args=args, namespace=namespace)

            if len(argument_parser._registries.get(AKU_DELAY, {})) == 0:
                break
            else:
                names = []
                for name, delay in list(argument_parser._registries[AKU_DELAY].items()):
                    names.append(name)
                    delay()

                for name in names:
                    del argument_parser._registries[AKU_DELAY][name]

        if self.add_help:
            argument_parser.add_argument(
                '--help', action='help', default=SUPPRESS,
                help='show this help message and exit',
            )
        for action in argument_parser._actions:
            if action.required is None:
                action.required = True
        return args, namespace

    def aku_parse_args(self, args: List[str] = None, namespace: Namespace = None):
        args, namespace = self._parse(args=args, namespace=namespace)
        return self.parse_args(args=args, namespace=namespace)

    def aku_parse_known_args(self, args: List[str] = None, namespace: Namespace = None):
        args, namespace = self._parse(args=args, namespace=namespace)
        return self.parse_known_args(args=args, namespace=namespace)

    def error(self, message: str) -> None:
        raise RuntimeError(message)

    def run(self, args: List[str] = None, namespace: Namespace = None):
        namespace = self.aku_parse_args(args=args, namespace=namespace)
        if isinstance(namespace, Namespace):
            namespace = namespace.__dict__

        partial, literal = {}, {}
        for key, value in namespace.items():

            partial_co = partial
            literal_co = literal
            *names, key = key.split('.')
            for name in names:
                partial_co = partial_co.setdefault(name, {})
                literal_co = literal_co.setdefault(name, {})
            if key == AKU_FN:
                partial_co[key], literal_co[key] = value
            else:
                partial_co[key] = literal_co[key] = value

        def recur_partial(item):
            if isinstance(item, dict):
                if AKU_FN in item:
                    func = item.pop(AKU_FN)
                    kwargs = {k: recur_partial(v) for k, v in item.items()}
                    return functools.partial(func, **kwargs)
                else:
                    return {k: recur_partial(v) for k, v in item.items()}
            else:
                return item

        def recur_literal(item):
            out, keys, values = {}, [], []

            def recur(prefixes, domain, v):
                nonlocal keys, values

                if isinstance(v, dict):
                    for x, y in v.items():
                        if x == AKU_FN:
                            out['-'.join((*prefixes[1:], domain.removesuffix('_')))] = y
                        elif domain.endswith('_'):
                            recur(prefixes + (domain.removesuffix('_'),), x, y)
                        else:
                            recur(prefixes, x, y)
                else:
                    out['-'.join(prefixes + (domain,))] = v

            recur((), '', item)
            return out

        partial = recur_partial(partial)

        assert len(partial) == 1
        for _, fn in partial.items():
            if inspect.getfullargspec(fn).varkw is None:
                return fn()
            else:
                return fn(**{AKU: recur_literal(literal)})
