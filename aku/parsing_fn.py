from typing import get_type_hints, Type, Callable, Any, Dict, Set, Union, List, Tuple

ParsingFn = Union[Callable[[str], Any], Type]
parsing_fn_registry: Dict[Any, ParsingFn] = {}

ParsingFnGen = Callable[[Any], ParsingFn]
parsing_fn_gen_registry: Dict[Any, ParsingFnGen] = {}


def register_parsing_fn(fn: ParsingFn) -> ParsingFn:
    retype = get_type_hints(fn)['return']
    assert retype not in parsing_fn_registry, \
        f'the parsing function of {retype} is already registered ' \
        f'({parsing_fn_registry[retype]})'

    parsing_fn_registry[retype] = fn
    return fn


def register_parsing_fn_gen(origin: Any):
    def _register_parsing_fn_gen(fn: ParsingFnGen) -> ParsingFnGen:
        assert origin not in parsing_fn_gen_registry, \
            f'the parsing function generator of {origin} is already registered ' \
            f'({parsing_fn_gen_registry[origin]})'

        parsing_fn_gen_registry[origin] = fn
        return fn

    return _register_parsing_fn_gen


def get_parsing_fn(retype: Type) -> ParsingFn:
    if retype in parsing_fn_registry:
        return parsing_fn_registry[retype]
    origin = getattr(retype, '__origin__', None)
    args, *_ = getattr(retype, '__args__', (None,))
    if origin in parsing_fn_gen_registry:
        return parsing_fn_gen_registry[origin](args)
    return retype


@register_parsing_fn
def str2bool(option_string: str) -> bool:
    option_string = option_string.strip().lower()
    if option_string in ('1', 't', 'true', 'y', 'yes'):
        return True
    if option_string in ('0', 'f', 'false', 'n', 'no'):
        return False
    raise ValueError(f'{option_string} is not a {bool} value.')


@register_parsing_fn
def str2none(option_string: str) -> type(None):
    option_string = option_string.strip().lower()
    if option_string in ('nil', 'null', 'none'):
        return True
    raise ValueError(f'{option_string} is not a {type(None)} value.')


@register_parsing_fn_gen(set)
def str2set(retype: Any, sep: str = ',') -> ParsingFn:
    @register_parsing_fn
    def _str2set(option_string: str) -> Set[retype]:
        option_string = option_string.strip()
        if option_string.startswith('{') and option_string.endswith('}'):
            return set(map(get_parsing_fn(retype), option_string[1:-1].split(sep)))
        raise ValueError(f'{option_string} is not a Set[{retype}] value.')

    return _str2set


@register_parsing_fn_gen(list)
def str2list(retype: Any, sep: str = ',') -> ParsingFn:
    @register_parsing_fn
    def _str2list(option_string: str) -> List[retype]:
        option_string = option_string.strip()
        if option_string.startswith('[') and option_string.endswith(']'):
            return list(map(get_parsing_fn(retype), option_string[1:-1].split(sep)))
        raise ValueError(f'{option_string} is not a List[{retype}] value.')

    return _str2list


@register_parsing_fn_gen(tuple)
def str2tuple(retype: Any, sep: str = ',') -> ParsingFn:
    @register_parsing_fn
    def _str2tuple(option_string: str) -> Tuple[retype, ...]:
        option_string = option_string.strip()
        if option_string.startswith('(') and option_string.endswith(')'):
            return tuple(map(get_parsing_fn(retype), option_string[1:-1].split(sep)))
        raise ValueError(f'{option_string} is not a Tuple[{retype}, ...] value.')

    return _str2tuple


if __name__ == '__main__':
    print(get_parsing_fn(List[List[int]])('[[23,23],[34]]'))
    # print(get_parsing_fn(List[Set[int]])('[{23,23},{34}]'))
