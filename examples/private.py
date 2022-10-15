from aku import Aku

app = Aku()


@app.register
def foo(_: int = 1, __a: int = 2, __b: int = 3, **kwargs):
    print(f'_ => {_}')
    print(f'__a => {__a}')
    print(f'__b => {__b}')
    print(kwargs['@aku'])


if __name__ == '__main__':
    app.run()
