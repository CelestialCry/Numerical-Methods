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

    def __init__(self, function = lambda x: 0, mi = 0, ma = 1):
        self.function, self.max_dom, self.min_dom = function, ma, mi
    
    def plot(self, start = None, end = None, step = 50):
        if start == None or end == None:
            xs = np.linspace(self.min_dom, self.max_dom, step)
            ys = list(map(self.function, xs))
        else:
            xs = np.linspace(start, end, step)
            ys = list(map(self.function, xs))

        plt.plot(xs, ys)

    def __call__(self, x):
        return self.function(x)


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

    def __call__(self, x):
        super().__call__(x)

class PiecewiseLagrange(Lagrange):
    __slots__ = ["functions", "interval"]

    def __init__(self, pinterval):
        self.points = pinterval
        self.functions = [Lagrange(plist) for plist in pinterval]
        self.interval = [(lambda x: (min(x), max(x)))(l.sep()[0]) for l in self.functions]
        self.function = lambda x: self.nfunction(x)
        self.min_dom, self.max_dom = self.interval[0][0], self.interval[-1][1]

    def sep(self):
        return [p.sep() for p in functions]

    def nfunction(self, x):
        if x < self.interval[0][1]:
            return self.functions[0](x)
        for i in range(1, len(self.interval)-1):
            (m, n) = self.interval[i]
            if m < x and x < n:
                return self.functions[i](x)
        return self.functions[-1](x)

    def __call__(self, x):
        super().__call__(x)

    def plot(self):
        super().plot()

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

class ErrorCompare(Plottable):
    __slots__ = ["sqErr", "supErr", "N"]

    def __init__(self, function, mi, ma, n = 10):
        super().__init__(function, mi, ma)
        #self.N = n
        #self.sqErr, self.supErr = [self.err2(m+1) for m in range(1, n)], [self.errSup(m+1) for m in range(1, n)]

    def genny(self, steps = None):
        return None

    def err2(self, n):
        p, f = [P["y"] for P in equiNode(self.min_dom, self.max_dom, 100*n, self.genny(steps = n).function)], [P["y"] for P in equiNode(self.min_dom, self.max_dom, 100*n, self.function)]
        return np.sqrt((self.max_dom-self.min_dom)/(100*n)*sum([(y-x)**2 for (x,y) in zip(p,f)]))

    def errSup(self, n):
        p, f = [P["y"] for P in equiNode(self.min_dom, self.max_dom, 100*n, self.genny(steps = n).function)], [P["y"] for P in equiNode(self.min_dom, self.max_dom, 100*n, self.function)]
        return max([abs(y-x) for (x,y) in zip(p,f)])

    def __call__(self, x):
        super().__call__(x)
    
    def plot(self):
        plt.plot(range(2, self.N+1), self.sqErr, label = "Square Error")
        plt.plot(range(2, self.N+1), self.supErr, label = "Sup Error")
        plt.legend()


class ErrorLagrange(ErrorCompare):
    def __init__(self, function, mi, ma, n = 10):
        super().__init__(function, mi, ma, n)
        self.N = n
        self.sqErr, self.supErr = [self.err2(m+1) for m in range(1, n)], [self.errSup(m+1) for m in range(1, n)]

    def genny(self, steps = None, ver = "Equi"):
        if steps == None:
            steps = self.N
        if ver == "Equi":
            return Lagrange(equiNode(self.min_dom, self.max_dom, steps, self.function))
        if ver == "Cheby":
            return Lagrange(chebyNode(self.min_dom, self.max_dom, steps, self.function))
        return None

    def err2(self, n):
        return super().err2(n)

    def errSup(self, n):
        return super().errSup(n)

    def __call__(self, x):
        super().__call__(x)
    
    def plot(self):
        super().plot()

class ErrorPiecewiseLagrange(ErrorCompare):
    __slots__ = ["K"]
    def __init__(self, function, mi, ma, n = 10, k = 10):
        super().__init__(function, mi, ma)
        self.N, self.K = n, k
        self.sqErr, self.supErr = None, None

    def genny(self, k):
        intervals = np.linspace(self.min_dom, self.max_dom, k)
        intervals = [(intervals[i], intervals[i+1]) for i in range(len(intervals)-1)]
        pintervals = [equiNode(mi, ma, self.N, self.function) for (mi, ma) in intervals]
        return PiecewiseLagrange(pintervals)

    def err2(self, k):
        return super().err2(k)

    def errSup(self, k):
        return super().errSup(k)

    def __call__(self, x):
        super().__call__(x)

    def plot(self):
        super().plot()

# This is supposed to be defined on [0,1]
def a(x):
    return np.cos(2*np.pi*x)

# This is supposed to be defined on [0,π/4]
def b(x):
    return np.exp(3*x)*np.sin(2*x)

"""
# Task i)
plt.figure()
r = Plottable(runge, -5, 5)
r.plot()
p = Lagrange(chebyNode(-5, 5, 10, runge))
p.plot()
plt.show()
"""

# Task ii) Something is horribly wrong here
#plt.figure()
u = ErrorPiecewiseLagrange(a, 0, 1)
print(u.genny(5)(0))
#u.plot()
#plt.show()

"""
plt.figure()
u = ErrorLagrange(b, 0, np.pi/4, 50)
u.plot()
plt.show()
"""