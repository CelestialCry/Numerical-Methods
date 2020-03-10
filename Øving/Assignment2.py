class Point:

    __slots__ = ["x", "y"]

    def __init__(self, x = 0, y = 0):
        self.x = x
        self.y = y

    def __repr__(self):
        return "(x:" + str(self.x) + "; " + "y:" + str(self.y) + ")"

    def __setitem__(self, place, val):
        if place == "x" or place == 0:
            self.x = val
        if place == "y" or place == 1:
            self.y = val
        else:
            raise IndexError(f"{place} is out of range")

    def __getitem__(self, place):
        if place == "x" or place == 0:
            return self.x
        if place == "y" or place == 1:
            return self.y
        else:
            raise IndexError(f"{place} is out of range")
    
class Lagrange():

    __slots__ = ["polynomial", "points"]

    def __init__(self, plist):
        self.points = plist
        xs, ys = self.sep()

    def sep(self):
        return [p["x"] for p in self.points], [p["y"] for p in points]

print([Point(2,3)])
