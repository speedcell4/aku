# aku

[![Build Status](https://travis-ci.org/speedcell4/aku.svg?branch=master)](https://travis-ci.org/speedcell4/aku)

setup your argument parser speedily

## Installation

```bash
python3.6 -m pip install git+https://github.com/speedcell4/aku.git --upgrade
```

## Usage

```python
# file test_single_function.py
import aku

app = aku.App()


@app.register
def add(a: int, b: int = 2):
    print(f'{a} + {b} => {a + b}')


app.run()
```

then `aku` will automatically add argument option according to your function signature.

```bash
~ python tests/test_single_function.py --help    
usage: aku [-h] --a A [--b B]

optional arguments:
  -h, --help  show this help message and exit
  --a A       a (default: None)
  --b B       b (default: 2)

```

if you registered more than one functions, then sub-parser will be utilized.

```python
# file test_multi_functions.py
import aku

app = aku.App()


@app.register
def add(a: int, b: int = 2):
    print(f'{a} + {b} => {a + b}')


@app.register
def say_hello(name: str):
    print(f'hello {name}')


app.run()
```

your argument parser interface will looks like,

```bash
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

~ python tests/test_multi_functions.py say_hello --name aku
hello aku
```

## TODOs

- [ ] docstring parsing
- [ ] `args` and `kwargs` analyzing
- [ ] more friendly `FormatterClass`
- [ ] dependency relationship among parameters
- [ ] unit tests and Travis CI
- [ ] register type converter function
- [ ] builder pattern, e.g. @then @final \[@when]
