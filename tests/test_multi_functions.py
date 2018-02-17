import aku

app = aku.App()


@app.register
def add(a: int, b: int = 2):
    print(f'{a} + {b} => {a + b}')


@app.register
def say_hello(name: str):
    print(f'hello {name}')


@app.register
def miao(nice: ['aaa', 'bbb']):
    print(nice)


if __name__ == '__main__':
    app.run()
