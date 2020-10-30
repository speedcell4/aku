# Aku

[![Actions Status](https://github.com/speedcell4/aku/workflows/unit-tests/badge.svg)](https://github.com/speedcell4/aku/actions)
[![PyPI version](https://badge.fury.io/py/aku.svg)](https://badge.fury.io/py/aku)
[![Downloads](https://pepy.tech/badge/aku)](https://pepy.tech/project/aku)

Aku is an interactive annotation-driven `ArgumentParser` generator.

## Requirements

* Python 3.7 or higher

## Install

```bash
python -m pip install aku --upgrade
```

## Usage

```python
from aku import Aku

aku = Aku()


@aku.option
def add(a: int, b: int = 2):
    print(f'{a} + {b} => {a + b}')


aku.run()
```

`aku` will automatically add argument options according to your function signature.

```shell script
python3 foo.py --help
usage: foo.py [-h] --a int [--b int]

optional arguments:
  -h, --help  show this help message and exit
  --a int     a
  --b int     b (default: 2)
```