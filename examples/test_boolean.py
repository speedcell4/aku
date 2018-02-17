import aku
from aku.annotations import boolean

app = aku.App()


@app.register
def logical_and(a: boolean, b: boolean = False):
    print(f'{a} and {b} => {a and b}')


if __name__ == '__main__':
    app.run()
