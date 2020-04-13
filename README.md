# Aku

[![Actions Status](https://github.com/speedcell4/aku/workflows/unit-tests/badge.svg)](https://github.com/speedcell4/aku/actions)

An Annotation-driven ArgumentParser Generator

## Requirements

Python 3.7 or higher

## Install

```bash
python -m pip install aku --upgrade
```

## Quick Start

```python
# tests/test_single_function.py
from aku import Aku

aku = Aku()


@aku.register
def add(a: int, b: int = 2):
    print(f'{a} + {b} => {a + b}')


aku.run()
```

`aku` will automatically add argument options according to your function signature.

```
~ python tests/test_single_function.py --help    
usage: aku [-h] --a A [--b B]

optional arguments:
  -h, --help  show this help message and exit
  --a A       a (default: None)
  --b B       b (default: 2)

```

Registering more than one function will make `aku` add them to sub-parser respectively (and lazily).

```python
# file test_multi_functions.py
from aku import Aku

aku = Aku()


@aku.register
def add(a: int, b: int = 2):
    print(f'{a} + {b} => {a + b}')


@aku.register
def say_hello(name: str):
    print(f'hello {name}')


aku.run()
```

Similarly, your command line interface will look like,

```
~ python tests/test_multi_functions.py --help    
usage: aku [-h] {add,say_hello} ...

positional arguments:
  {add,say_hello}

optional arguments:
  -h, --help       show this help message and exit

~ python tests/test_multi_functions.py say_hello --help
usage: aku say_hello [-h] --name NAME

optional arguments:
  -h, --help   show this help message and exit
  --name NAME  name (default: None)
```
