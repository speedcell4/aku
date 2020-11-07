from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, SUPPRESS
from pathlib import Path


def tp_bool(arg_strings: str) -> bool:
    arg_strings = arg_strings.lower().strip()
    if arg_strings in ('t', 'true', 'y', 'yes', '1'):
        return True
    if arg_strings in ('f', 'false', 'n', 'no', '0'):
        return False
    raise ValueError


argument_parser = ArgumentParser(
    formatter_class=ArgumentDefaultsHelpFormatter,
)
argument_parser.add_argument('--a', type=int, metavar='int', default=SUPPRESS, required=True, help='a')
argument_parser.add_argument('--b', type=tp_bool, metavar='bool', default=True, help='b')
argument_parser.add_argument('--c', type=str, metavar='str', default='3', help='c')
argument_parser.add_argument('--d', type=float, metavar='float', default=4.0, help='d')
argument_parser.add_argument('--e', type=Path, metavar='path', default=Path.home(), help='e')

args = argument_parser.parse_args().__dict__
for key, value in args.items():
    print(f'{key} => {value}')
