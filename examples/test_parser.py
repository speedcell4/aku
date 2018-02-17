import pathlib

from aku.annotations import Path, boolean


class Parser(object):
    def __init__(self, nb_word: int, path: Path(ensure=True)):
        self.nb_word = nb_word
        self.path = path


def run_parser(
        nb_word: int,
        char_embedding: boolean, word_embedding: boolean,
        model: Path(ensure=True, expanduser=True, absolute=True) = None):
    if char_embedding:
        pathlib.Path(f'{path}/{home}')
    if word_embedding:
        pathlib.Path(f'{path}/{home}')
