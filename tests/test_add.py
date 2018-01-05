import argumentize

app = argumentize.App()


@app.register
def add(a: int, b: int = 2):
    print(f'{a} + {b} => {a + b}')
    return a + b


app.run()
