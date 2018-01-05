import argumentize

app = argumentize.App()


@app.register
def add(a: int, b: int = 2):
    print(f'{a} + {b} => {a + b}')


@app.register
def sub(a: int, b: int = 3):
    print(f'{a} - {b} => {a - b}')


app.run()
