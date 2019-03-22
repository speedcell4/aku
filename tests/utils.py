class Point(object):
    def __init__(self, x: int, y: int) -> None:
        self.x = x
        self.y = y


class Circle(object):
    def __init__(self, center: Point, radius: int) -> None:
        self.center = center
        self.radius = radius


