from argparse import Action, ArgumentParser, Namespace
from typing import Any, Optional

from aku import annotation as ty
from aku.annotation import annotation_iter

DONE = '_DONE'


def get_name(name, prefix):
    if prefix is None:
        return name
    return f'{prefix}_{name}'


def cat_prefix(prefix1, prefix2):
    if prefix1 is None:
        return prefix2
    if prefix2 is None:
        return prefix1
    return f'{prefix1}_{prefix2}'


def add_primitive(parser: ArgumentParser, prefix: str, name: str, annotation: Any, default: Any, delays):
    dest = get_name(name, prefix)
    parser.add_argument(
        f'--{dest}', dest=dest, help=name, default=default,
        type=ty.primitive_type(annotation), metavar=ty.primitive_metavar(annotation),
    )
    return dest


class ListAction(Action):
    def __call__(self, action_parser: ArgumentParser, namespace: Namespace, values, option_string) -> None:
        setattr(namespace, self.dest, getattr(self, self.dest, []) + [values])


def add_list(parser: ArgumentParser, prefix: str, name: str, annotation: Any, default: Any, delays):
    dest = get_name(name, prefix)

    parser.add_argument(
        f'--{dest}', dest=dest, help=name, default=default,
        type=ty.list_type(annotation), metavar=ty.list_metavar(annotation),
        action=ListAction, required=True,
    )

    return dest


class TupleAction(Action):
    def __call__(self, action_parser: ArgumentParser, namespace: Namespace, values, option_string) -> None:
        setattr(namespace, self.dest, getattr(self, self.dest, ()) + (values,))


def add_tuple(parser: ArgumentParser, prefix: str, name: str, annotation: Any, default: Any, delays):
    dest = get_name(name, prefix)

    parser.add_argument(
        f'--{dest}', dest=dest, help=name, default=default,
        type=ty.tuple_type(annotation), metavar=ty.tuple_metavar(annotation),
        action=TupleAction, required=True,
    )

    return dest


def add_value_union(parser: ArgumentParser, prefix: str, name: str, annotation: Any, default: Any, delays):
    dest = get_name(name, prefix)

    parser.add_argument(
        f'--{dest}', dest=dest, help=name, default=default,
        type=ty.value_union_type(annotation), metavar=ty.value_union_metavar(annotation),
        choices=annotation,
    )

    return dest


def add_type_union(parser: ArgumentParser, prefix: str, name: str, annotation: Any, default: Any, delays):
    dest = get_name(name, prefix)
    obj_dest = f'@{dest}'
    fn_mapping = {
        a.__name__: a
        for a in ty.type_union_args(annotation)
    }

    class TypeUnionAction(Action):
        def __call__(self, action_parser: ArgumentParser, namespace: Namespace, values, option_string) -> None:
            if not getattr(self, DONE, False):
                setattr(self, DONE, True)
                setattr(namespace, obj_dest, values)

                func = fn_mapping[values]
                add_function(
                    parser=parser, prefix=prefix,
                    name=dest, annotation=func, default=None, delays=delays,
                )

                action_parser.set_defaults(**{dest: func})
                action_parser.parse_known_args(namespace=namespace)

    parser.add_argument(
        f'--{dest}', dest=obj_dest, help=name, default=default if isinstance(default, str) else default.__name__,
        type=ty.type_union_type(annotation), metavar=ty.type_union_metavar(annotation),
        action=TypeUnionAction, choices=tuple(fn_mapping.keys()),
    )

    return dest


def add_type_var(parser: ArgumentParser, prefix: str, name: str, annotation: Any, default: Any, delays):
    dest = get_name(name, prefix)
    obj_dest = f'@{dest}'
    fn_mapping = {
        a.__name__: a
        for a in ty.type_var_args(annotation)
    }

    class TypeVarAction(Action):
        def __call__(self, action_parser: ArgumentParser, namespace: Namespace, values, option_string) -> None:
            if not getattr(self, DONE, False):
                setattr(self, DONE, True)
                setattr(namespace, obj_dest, values)

                func = fn_mapping[values]
                add_function(
                    parser=parser, prefix=cat_prefix(prefix, annotation.__name__),
                    name=dest, annotation=func, default=None, delays=delays,
                )

                action_parser.set_defaults(**{dest: func})
                action_parser.parse_known_args(namespace=namespace)

    parser.add_argument(
        f'--{dest}', dest=obj_dest, help=name, default=default if isinstance(default, str) else default.__name__,
        type=ty.type_union_type(annotation), metavar=ty.type_union_metavar(annotation),
        action=TypeVarAction, choices=tuple(fn_mapping.keys()),
    )

    return dest


def add_function(parser: ArgumentParser, prefix: Optional[str], name: Optional[str], annotation: Any, default: Any,
                 delays):
    for nn, aa, dd in annotation_iter(annotation):
        if ty.is_list(aa):
            add_op = add_list
        elif ty.is_tuple(aa):
            add_op = add_tuple
        elif ty.is_value_union(aa):
            add_op = add_value_union
        elif ty.is_type_union(aa):
            add_op = add_type_union
        elif ty.is_type_var(aa):
            add_op = add_type_var
        elif ty.is_callable(aa):
            add_op, aa = add_function, dd
        else:
            add_op = add_primitive

        key = add_op(
            parser=parser, prefix=prefix,
            delays=delays, name=nn, annotation=aa, default=dd,
        )
        if name is not None:
            delays.append((name, nn, key))
