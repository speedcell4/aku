from argparse import Action, ArgumentParser, Namespace

from aku.utils import AKU_VISITED


class StoreAction(Action):
    def __call__(self, parser: ArgumentParser, namespace: Namespace, values, option_string=None):
        setattr(namespace, self.dest, values)
        self.required = False


class AppendListAction(Action):
    def __call__(self, parser: ArgumentParser, namespace: Namespace, values, option_string=None):
        if not getattr(self, AKU_VISITED, False):
            setattr(self, AKU_VISITED, True)
            setattr(namespace, self.dest, [])
        getattr(namespace, self.dest).append(values)
        self.required = False
