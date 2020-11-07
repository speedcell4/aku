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

## Usage

The key idea of aku to generate `ArgumentParser` according to the type annotations of functions. For example, to register single function with only primitive types, i.e., `int`, `bool`, `str`, `float`, `Path`, etc.

```python
from pathlib import Path

from aku import Aku

aku = Aku()


@aku.option
def foo(a: int, b: bool = True, c: str = '3', d: float = 4.0, e: Path = Path.home()):
    print(f'a => {a}')
    print(f'b => {b}')
    print(f'c => {c}')
    print(f'd => {d}')
    print(f'e => {e}')


aku.run()
```

`aku` will generate a `ArgumentParser` which provides your command line interface looks like below,

```shell script
~ python examples/foo.py --help 
usage: foo.py [-h] --a int [--b bool] [--c str] [--d float] [--e path]

optional arguments:
  -h, --help  show this help message and exit
  --a int     a
  --b bool    b (default: True)
  --c str     c (default: 3)
  --d float   d (default: 4.0)
  --e path    e (default: /Users/home)
```

Of course you can achieve the same functions by instantiating an `ArgumentParser`, but `aku` certainly makes such steps simple and efficient.

```python
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, SUPPRESS
from pathlib import Path


def tp_bool(arg_strings: str) -> bool:
    arg_strings = arg_strings.lower().strip()
    if arg_strings in ('t', 'true', 'y', 'yes', '1'):
        return True
    if arg_strings in ('f', 'false', 'n', 'no', '0'):
        return False
    raise ValueError


argument_parser = ArgumentParser(
    formatter_class=ArgumentDefaultsHelpFormatter,
)
argument_parser.add_argument('--a', type=int, metavar='int', default=SUPPRESS, required=True, help='a')
argument_parser.add_argument('--b', type=tp_bool, metavar='bool', default=True, help='b')
argument_parser.add_argument('--c', type=str, metavar='str', default='3', help='c')
argument_parser.add_argument('--d', type=float, metavar='float', default=4.0, help='d')
argument_parser.add_argument('--e', type=Path, metavar='path', default=Path.home(), help='e')

args = argument_parser.parse_args().__dict__
for key, value in args.items():
    print(f'{key} => {value}')
```

Moreover, if you register more than one functions, e.g., register function `add`,

```python
@aku.option
def add(x: int, y: int):
    print(f'{x} + {y} => {x + y}')
```

Then you can choose which one to run by passing its name as the first parameter,

```shell script
~ python examples/bar.py foo --help
usage: bar.py foo [-h] --a int [--b bool] [--c str] [--d float] [--e path]

optional arguments:
  -h, --help  show this help message and exit
  --a int     a
  --b bool    b (default: True)
  --c str     c (default: 3)
  --d float   d (default: 4.0)
  --e path    e (default: /Users/home)

~ python examples/bar.py add --help
usage: bar.py add [-h] --x int --y int

optional arguments:
  -h, --help  show this help message and exit
  --x int     x
  --y int     y

~ python examples/bar.py add --x 1 --y 2
1 + 2 => 3
```