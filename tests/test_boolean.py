import argumentize
from argumentize.annotations import boolean

app = argumentize.App()


@app.register
def logical_and(a: boolean, b: boolean = False):
    print(f'{a} and {b} => {a and b}')


app.run()
