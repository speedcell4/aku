from argparse import ArgumentParser


def fetch_actions(argument_parser: ArgumentParser) -> str:
    msg = ', '.join([
        action.option_strings[-1]
        for action in argument_parser._actions
    ])
    return f"[{msg}]"
