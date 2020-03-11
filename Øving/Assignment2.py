import matplotlib.pyplot as plt
import autograd.numpy as np
from autograd import jacobian, grad
from functools import reduce, partial

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

class Plottable():
    
    __slots__ = ["function"]

    def __init__(self, function = lambda x: 0):
        self.function = function

    def plot(self, start, end):
        xs = np.linspace(start, end)
        ys = list(map(lambda x: self.function(x), xs))

        plt.figure()
        plt.plot(xs, ys)
        plt.show()
    
class Lagrange(Plottable):

    __slots__ = ["points", "max_dom", "min_dom"]

    #Merkelig nok virker denne dritten med autograd!
    def __init__(self, plist):
        self.points = plist
        xs, ys = self.sep()
        self.max_dom, self.min_dom = max(xs), min(xs)
        λj = lambda xj, ls, x: reduce(lambda a,b: a*b, map(partial(lambda y, yj, arg: (y-arg)/(yj-arg), x, xj), ls))
        self.function = lambda x: sum([ys[i]*partial(λj, xs[i], xs[0:i] + xs[i+1:len(xs)])(x) for i in range(len(xs))])

    def sep(self):
        return [p["x"] for p in self.points], [p["y"] for p in self.points] 

    def plot(self, start = None, end = None):
        if start == None or end == None:
            super().plot(self.min_dom, self.min_dom)
        else:
            super().plot(start, end)

pol = Lagrange([Point(-1,1), Point(1,1), Point(0,0)])
print(jacobian(pol.function)(3.0))