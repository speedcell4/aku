from argparse import ArgumentDefaultsHelpFormatter
from argparse import SUPPRESS

from aku.utils import AKU_FN


class AkuFormatter(ArgumentDefaultsHelpFormatter):
    def __init__(self, prog: str, indent_increment: int = 2, max_help_position: int = 60, width: int = None) -> None:
        super(AkuFormatter, self).__init__(
            prog=prog, indent_increment=indent_increment,
            max_help_position=max_help_position, width=width,
        )

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
