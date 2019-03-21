class ValueUnion(type):
    def __class_getitem__(cls, values):
        return values


if __name__ == '__main__':
    print(ValueUnion[1, 2, 3])

