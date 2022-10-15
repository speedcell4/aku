# Aku

[![Actions Status](https://github.com/speedcell4/aku/workflows/unit-tests/badge.svg)](https://github.com/speedcell4/aku/actions)
[![PyPI version](https://badge.fury.io/py/aku.svg)](https://badge.fury.io/py/aku)
[![Downloads](https://pepy.tech/badge/aku)](https://pepy.tech/project/aku)

An interactive annotation-driven `ArgumentParser` generator

## Requirements

* Python 3.7 or higher

## Install

```shell script
python -m pip install aku --upgrade
```

## Type Annotations

* Primitive types, e.g., `int`, `str`, `bool`, `float`, `Path`.
* Container types,
    - List, i.e., `List[T]`
    - Tuple, i.e., homogeneous `Tuple[T, ...]` and heterogeneous `Tuple[T1, T2, T3]`
    - Set and FrozenSet, i.e., `Set[T]` and `FrozenSet[T]`
    - Literal, e.g., `Literal[42, 1905]`
* Nested types
    - Function, e.g., `Type[F]`
    - Union of functions, e.g., `Union[Type[F1], Type[F2], Type[F3]]`

## Namespace

* Single leading underline, i.e., `_name`, means omitting this argument from `literal`, while its nested arguments will be kept
* Double leading underlines, i.e., `__name`, remove this argument and all of its nested arguments
* Single tailing underline, i.e., `name_`, opens a new namespace for its nested arguments.