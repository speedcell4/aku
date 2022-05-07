from argparse import Action, ArgumentParser, Namespace


class StoreAction(Action):
    def __call__(self, parser: ArgumentParser, namespace: Namespace, values, option_string=None):
        setattr(namespace, self.dest, values)
        self.required = False


class AppendListAction(Action):
    AKU_VISITED = '_aku_visited'

    def __call__(self, parser: ArgumentParser, namespace: Namespace, values, option_string=None):
        if not getattr(self, self.AKU_VISITED, False):
            setattr(self, self.AKU_VISITED, True)
            setattr(namespace, self.dest, [])
        getattr(namespace, self.dest).append(values)
        self.required = False
