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
    __slots__ = ["function", "max_dom", "min_dom"]

    def __init__(self, function = lambda x: 0, ma = 0, mi = 1):
        self.function, self.max_dom, self.min_dom = function, ma, mi
    
    def plot(self, start = None, end = None, step = 50):
        if start == None or end == None:
            xs = np.linspace(self.min_dom, self.max_dom, step)
            ys = list(map(self.function, xs))
        else:
            xs = np.linspace(start, end, step)
            ys = list(map(self.function, xs))

        plt.plot(xs, ys)


class Lagrange(Plottable):
    __slots__ = ["points"]

    #Merkelig nok virker denne dritten med autograd!
    def __init__(self, plist):
        self.points = plist
        xs, ys = self.sep()
        self.max_dom, self.min_dom = max(xs), min(xs)
        λj = lambda xj, ls, x: reduce(lambda a,b: a*b, map(partial(lambda y, yj, arg: (y-arg)/(yj-arg), x, xj), ls))
        self.function = lambda x: sum([ys[i]*partial(λj, xs[i], xs[0:i] + xs[i+1:len(xs)])(x) for i in range(len(xs))])

    def sep(self):
        return [p["x"] for p in self.points], [p["y"] for p in self.points] 

    def plot(self, step = 50):
        super().plot(self.min_dom, self.max_dom, step)

def equiNode(start, end, step, f = (lambda x: 0)):
    xs = np.linspace(start, end, step)
    ys = map(f, xs)
    return [Point(x,y) for (x,y) in zip(xs,ys)]

def chebyNode(start, end, steps, f = lambda x: 0):
    xs = [(end-start)/2*(np.cos(np.pi*(2*x+1)/(2*steps)))+(end+start)/2 for x in range(steps)]
    ys = map(f, xs)
    return [Point(x,y) for (x,y) in zip(xs, ys)]

def runge(x):
    return 1/(x**2+1)

print(len(chebyNode(-5, 5, 10, runge)))

plt.figure()

r = Plottable(runge, -5, 5)
r.plot()

p = Lagrange(chebyNode(-5, 5, 10, runge))
p.plot()

plt.show()
