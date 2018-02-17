import aku

app = aku.App()


@app.register
def add(a: int, b: int = 2):
    print(f'{a} + {b} => {a + b}')


if __name__ == '__main__':
    app.run()
