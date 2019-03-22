from typing import Optional

from aku.utils import is_homo_tuple, is_list, is_optional, is_union, is_value_union, \
    unwrap_homo_tuple, unwrap_list, unwrap_optional, unwrap_union


def render_type(retype) -> Optional[str]:
    if is_optional(retype):
        args = render_type(unwrap_optional(retype))
        return f'{args}?'
    if is_union(retype):
        args = ', '.join(render_type(a) for a in unwrap_union(retype))
        return f'{{{args}}}'
    if is_list(retype):
        args = render_type(unwrap_list(retype))
        return f'[{args}]'
    if is_homo_tuple(retype):
        args = render_type(unwrap_homo_tuple(retype))
        return f'({args})'
    if is_value_union(retype):
        return None

    return f'{retype.__name__}'.capitalize()
