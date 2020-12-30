import functools
import inspect
import sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, Namespace, SUPPRESS
from typing import Type

from aku.tp import AkuTp
from aku.utils import _init_argument_parser, fetch_name, AKU, AKU_FN, AKU_ROOT


class AkuFormatter(ArgumentDefaultsHelpFormatter):
    def _format_actions_usage(self, actions, groups):
        required_option_strings = [
            action.option_strings[-1][2:]
            for action in actions if action.required
        ]
        if len(required_option_strings) > 0:
            return f'-- [{"|".join(required_option_strings)}]'
        return ''


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
                 allow_abbrev=False) -> None:
        super(Aku, self).__init__(
            prog=prog, usage=usage, description=description, epilog=epilog,
            parents=parents, formatter_class=formatter_class, prefix_chars=prefix_chars,
            fromfile_prefix_chars=fromfile_prefix_chars, argument_default=argument_default,
            conflict_handler=conflict_handler, add_help=False, allow_abbrev=allow_abbrev,
        )
        _init_argument_parser(self)

        self.options = []
        self.add_help = add_help

    def option(self, fn):
        self.options.append(fn)
        return fn

    def parse_aku(self, args=None) -> Namespace:
        assert len(self.options) > 0

        if args is None:
            args = sys.argv[1:]

        namespace, argument_parser = None, self
        if len(self.options) == 1:
            option = self.options[0]
            AkuTp[Type[option]].add_argument(
                argument_parser=argument_parser,
                name=AKU_ROOT, default=SUPPRESS,
                prefixes=(), domain=(),
            )
        else:
            subparsers = argument_parser.add_subparsers()
            options = {}
            for option in self.options:
                name = fetch_name(option)
                if name not in options:
                    options[name] = (option, subparsers.add_parser(name=name))
                else:
                    raise ValueError(f'{name} was already registered')

            if len(args) > 0 and args[0] in options:
                arg, *args = args
                option, argument_parser = options[arg]
                AkuTp[Type[option]].add_argument(
                    argument_parser=argument_parser,
                    name=AKU_ROOT, default=SUPPRESS,
                    prefixes=(), domain=(),
                )

        while True:
            namespace, args = argument_parser.parse_known_args(args=args, namespace=namespace)

            if len(argument_parser._registries['delay'][AKU]) == 0:
                break
            else:
                for delay in argument_parser._registries['delay'][AKU]:
                    delay()
                argument_parser._registries['delay'][AKU].clear()

        if self.add_help:
            argument_parser.add_argument(
                '--help', action='help', default=SUPPRESS,
                help='show this help message and exit',
            )
        for action in argument_parser._actions:
            if action.required is None:
                action.required = True
        return argument_parser.parse_args(args=args, namespace=namespace)

    def error(self, message: str) -> None:
        raise RuntimeError(message)

    def run(self, namespace: Namespace = None):
        if namespace is None:
            namespace = self.parse_aku()
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

        def recur_curry(item):
            if isinstance(item, dict):
                if AKU_FN in item:
                    func = item.pop(AKU_FN)
                    kwargs = {k: recur_curry(v) for k, v in item.items()}
                    return functools.partial(func, **kwargs)
                else:
                    return {k: recur_curry(v) for k, v in item.items()}
            else:
                return item

        def flatten_literal(item):
            keys, values = [], []

            def recur(k, v):
                nonlocal keys, values

                if isinstance(v, dict):
                    for x, y in v.items():
                        recur(k + (x,), y)
                else:
                    keys.append(k)
                    values.append(v)

            recur((), item)
            return {
                '-'.join([x[:-1] for x in k[1:-1] if x.endswith('_')] + [k[-1]]): v
                for k, v in zip(keys, values)
            }

        curry = recur_curry(curry)
        literal = flatten_literal(literal)

        assert len(curry) == 1
        for _, fn in curry.items():
            if inspect.getfullargspec(fn).varkw is None:
                return fn()
            else:
                return fn(**{AKU: literal})
