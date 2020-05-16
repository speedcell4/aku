from argparse import ArgumentParser, Namespace
from typing import List


class Aku(ArgumentParser):
    def parse_known_args(self, args: List[str] = None, namespace: Namespace = None):
        last_actions_len = -1
        while last_actions_len != len(self._actions):
            last_actions_len = len(self._actions)
            namespace, args = super(Aku, self).parse_known_args(args=args, namespace=namespace)
        return super(Aku, self).parse_known_args(args=args, namespace=namespace)

    def parse_args(self, args: List[str] = None, namespace: Namespace = None):
        namespace, args = super(Aku, self).parse_known_args(args=args, namespace=namespace)
        return super(Aku, self).parse_args(args=args, namespace=namespace)
