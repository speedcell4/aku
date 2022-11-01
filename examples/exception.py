from aku import Aku

aku = Aku()


@aku.register
def empty(x):
    print(f'x => {x}')


if __name__ == '__main__':
    aku.run()
