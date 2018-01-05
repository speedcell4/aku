# argumentize

setup your argument parser rapidly

## Installation

```bash
python3.6 -m pip install https://github.com/speedcell4/argumentize.git --upgrade
```

## Usage

```python
# file test_single_function.py
import argumentize

app = argumentize.App()


@app.register
def add(a: int, b: int = 2):
    print(f'{a} + {b} => {a + b}')


app.run()
```

then `argumentize` will automatically add argument option according to your function signature.

```bash
~ python tests/test_single_function.py --help    
usage: argumentize [-h] --a A [--b B]

optional arguments:
  -h, --help  show this help message and exit
  --a A       a (default: None)
  --b B       b (default: 2)

```

if you registered more than one functions, then sub-parser will be utilized.

```python
# file test_multi_functions.py
import argumentize

app = argumentize.App()


@app.register
def add(a: int, b: int = 2):
    print(f'{a} + {b} => {a + b}')


@app.register
def sub(a: int, b: int = 3):
    print(f'{a} - {b} => {a - b}')


app.run()
```

your argument parser interface will looks like,

```bash
~ python tests/test_multi_functions.py --help    
usage: argumentize [-h] {add,say_hello} ...

positional arguments:
  {add,say_hello}

optional arguments:
  -h, --help       show this help message and exit

~ python tests/test_multi_functions.py say_hello --help
usage: argumentize say_hello [-h] --name NAME

optional arguments:
  -h, --help   show this help message and exit
  --name NAME  name (default: None)

~ python tests/test_multi_functions.py say_hello --name argumentize
hello argumentize
```

## TODOs

- [ ] docstring parsing
- [ ] `args` and `kwargs` analyzing
- [ ] more friendly `FormatterClass`