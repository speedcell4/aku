import os


def get_max_help_position() -> int:
    try:
        return os.get_terminal_size().columns
    except OSError:
        return 24