import functools
import inspect
import sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, Namespace, SUPPRESS
from typing import Type

from aku.tp import AkuTp
from aku.utils import _init_argument_parser, fetch_name, AKU_FN, AKU, AKU_DELAY


class AkuFormatter(ArgumentDefaultsHelpFormatter):
    def _expand_help(self, action):
        params = dict(vars(action), prog=self._prog)
        if params['dest'].endswith(AKU_FN) and isinstance(params['default'], tuple):
            params['default'] = params['default'][1]
        for name in list(params):
            if params[name] is SUPPRESS:
                del params[name]
        for name in list(params):
            if hasattr(params[name], '__name__'):
                params[name] = params[name].__name__
        if params.get('choices') is not None:
            choices_str = ', '.join([str(c) for c in params['choices']])
            params['choices'] = choices_str
        return self._get_help_string(action) % params

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
                 always_use_subparse=False,
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
        self.always_use_subparse = always_use_subparse

    def option(self, fn):
        self.options.append(fn)
        return fn

    def parse_aku(self, args=None) -> Namespace:
        assert len(self.options) > 0

        if args is None:
            args = sys.argv[1:]

        namespace, argument_parser = None, self
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

        def abbreviate_literal(item):
            out, keys, values = {}, [], []

            def recur(prefixes, k, v):
                nonlocal keys, values

                if isinstance(v, dict):
                    for x, y in v.items():
                        if x == AKU_FN:
                            out['-'.join(prefixes[1:] + (k,))] = y
                        elif k.endswith('_'):
                            recur(prefixes + (k[:-1],), x, y)
                        else:
                            recur(prefixes, x, y)
                else:
                    out['-'.join(prefixes[1:] + (k,))] = v

            recur((), '', item)
            return out

        curry = recur_curry(curry)
        literal = abbreviate_literal(literal)

        assert len(curry) == 1
        for _, fn in curry.items():
            if inspect.getfullargspec(fn).varkw is None:
                return fn()
            else:
                return fn(**{AKU: literal})
