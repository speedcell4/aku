<div align="center">

# Aku

![GitHub Workflow Status (with event)](https://img.shields.io/github/actions/workflow/status/speedcell4/aku/unit-tests.yml?cacheSeconds=0)
![PyPI - Version](https://img.shields.io/pypi/v/aku?label=pypi%20version&cacheSeconds=0)
![PyPI - Downloads](https://img.shields.io/pypi/dm/aku?cacheSeconds=0)

</div>

An interactive annotation-driven `ArgumentParser` generator

## Installation

`python -m pip aku`

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
