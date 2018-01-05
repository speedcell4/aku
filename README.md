# argumentize

setup your argument parser rapidly

## Installation

```bash
python3.6 -m pip install https://github.com/speedcell4/argumentize.git --upgrade
```

## Usage

```python
# file test_add.py
import argumentize

app = argumentize.App()


@app.register
def add(a: int, b: int = 2):
    return a + b


app.run()
```

then `argumentize` will automatically add argument option according to your function signature.

```bash
~ python3.6 test_add.py --help 
usage: argumentize [-h] --a A [--b B]

optional arguments:
  -h, --help  show this help message and exit
  --a A       a (default: None)
  --b B       b (default: 2)

```