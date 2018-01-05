import argumentize

app = argumentize.App()


@app.register
def add(a: int, b: int = 2):
    print(f'{a} + {b} => {a + b}')


@app.register
def say_hello(name: str):
    print(f'hello {name}')


app.run()
