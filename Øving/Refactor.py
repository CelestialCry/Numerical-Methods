import time
import functools as func
import matplotlib.pyplot as plt
import autograd as aut
import autograd.numpy as np
from autograd.builtins import list, dict, tuple
from autograd import grad, jacobian
from math import floor, factorial
from operator import mul

#Binary Search
#-----
def binarySearch(array, searchFor):
    minIndex = 0
    m = len(array) - 1
    maxIndex = m
    midIndex = floor((minIndex + maxIndex)/2)

    while maxIndex >= minIndex:
        midIndex = floor((minIndex + maxIndex)/2)
        
        v = array[midIndex]

        if searchFor < v:
            maxIndex = midIndex - 1

        elif searchFor > v:
            minIndex = midIndex + 1

        elif searchFor == v:
            return midIndex

    if v <= searchFor:
        return midIndex
    else:
        return midIndex-1

#Plottable
#------

class Plottable():
    __slots__ = ["function", "start", "end", "history", "gradient"]

    def __init__(self, function=lambda x: 0, mi=0, ma=1, gradient = None):
        self.function, self.start, self.end, self.history = function, mi, ma, []
        if gradient == None:
            self.gradient = self.diff()
        else:
            self.gradient = gradient

    def plot(self, *args, start=None, end=None, step=500, **kwargs):
        if start == None or end == None:
            xs = np.linspace(self.end, self.start, step)
            ys = list(map(self.function, xs))
        else:
            xs = np.linspace(start, end, step)
            ys = list(map(self.function, xs))
        plt.plot(xs, ys, *args, **kwargs)

    def __repr__(self):  # Class representation
        plt.figure()
        self.plot()
        plt.show()
        return "printed!"

    def __call__(self, *args):  # Function calling overloading
        return self.function(*args)

    def diff(self):
        return grad(self.function)

# Gradient Descent
#-----
def gradientDescent(F, x0, γ = 10, ρ = 0.8, σ = 1.7, TOL = 1e-7, maxIter = 100):
    gradF = F.gradient
    x1 = x0
    φ = F(x1)
    F.history.append(φ)
    for m in range(maxIter):
        # χ = x0
        g = gradF(x1)
        for n in range(200):
            x1 = x0 - 1/γ*g
            ψ = F(x1)
            if ψ <= φ + np.dot(g,x1-x0)+γ/2*np.linalg.norm(x1-x0)**2:
                x0, φ, γ = x1, ψ, ρ*γ
                break
            else:
                γ *= σ
        F.history.append(φ)
        if np.linalg.norm(g) <= TOL: # or np.linalg.norm(g) <= TOL:
            break
    return x1

# NodesSpreads
#-------

def equiX(a,b,N):
    return np.linspace(a,b,N)

def equiNode(start, end, step, f = lambda x: 0):
    xs = np.linspace(start, end, step)
    ys = map(f, xs)
    return [a for a in zip(xs, ys)]

def chebyX(a,b,N):
    return [(b-a)/2*(np.cos(np.pi*(2*x+1)/(2*N))) + (b+a)/2 for x in range(N)]

def chebyNode(start, end, steps, f = lambda x: 0):
    xs = [(end - start) / 2 * (np.cos(np.pi * (2 * x + 1) / (2 * steps))) + (end + start) / 2 for x in range(steps)]
    ys = map(f, xs)
    return [(x, y) for (x, y) in zip(xs, ys)]

#Lagrange Polynomial
#-----

def lagrangify(ps):
    if len(ps) == 1:
        return lambda x : ps[0][1]
    else:
        λj = lambda j, ls, x: ps[j][1] * func.reduce(mul, [1] + [(x - arg) / (ls[j] - arg) for arg in ls if ls[j] != arg])
        return lambda x : sum([λj(i, [χ for χ,_ in ps], x) for i in range(len(ps))])

class LagrangePol(Plottable):

    def __init__(self, ps):
        xs = [x for x,_ in ps]
        start, end = min(xs), max(xs)
        function = lagrangify(ps)
        super().__init__(function, start, end, gradient = None)

#Spline Interpol
#-----

def splinify(ips):
    intervals = [x for x,_ in [min(ps) for ps in ips]]
    functions = [lagrangify(ps) for ps in ips]

    def pwf(x):
        i = binarySearch(intervals, x)
        return functions[i](x)
    return pwf

class SplinePol(Plottable):

    def __init__(self, ips):
        function = splinify(ips)
        minxs, maxxs = [x for x,_ in ips[0]], [x for x,_ in ips[-1]]
        start, end = min(minxs), max(maxxs)
        super().__init__(function, start, end, gradient = None)

#Optimized Lagrange Polynomial
#-----

def optLagrangify(psKnown, xs):

    ls = [x for x,_ in psKnown]
    a, b, N = min(ls), max(ls), len(ls)

    def lap(psKnown = psKnown):
        xsKnown, ysKnown = [], []
        for x,y in psKnown:
            xsKnown.append(x)
            ysKnown.append(y)
        def funkyfy(x):
            i = binarySearch(ls, x)
            if i >= N-1:
                i = N-2
            a = (ysKnown[i+1]-ysKnown[i])/(xsKnown[i+1]-xsKnown[i])
            return a*x+ysKnown[i]-a*xsKnown[i]
        return lambda x: funkyfy(x)

    f = lap()

    def cost(ks):
        ps = [(x,f(x)) for x in ks]
        k = (b-a)/N
        s = 0
        p = lagrangify(ps)
        for x,y in psKnown:
            s = s + (y-p(x))**2
        return k*s

    def lagGrad(i, ks, x):
        partOne = grad(f)(ks[i])*func.reduce(mul, [1] + [(x - ks[k])/(ks[i]-ks[k]) for k in range(len(ks)) if i != k])
        partTwo = f(ks[i])*sum([(ks[k]-x)/((ks[i]-ks[k])**2)*func.reduce(mul, [1] + [(x-ks[j])/(ks[i]-ks[j]) for j in range(len(ks)) if j != i and j != k]) for k in range(len(ks)) if k != i])
        partThree = sum([f(ks[k])*(x-ks[k])/((ks[k]-ks[i])**2)*func.reduce(mul, [1] + [(x-ks[j])/(ks[k]-ks[j]) for j in range(len(ks)) if k != j and j != i]) for k in range(len(ks)) if k != i])
        return partOne + partTwo + partThree

    def gradient(ks):
        ps = [(x,f(x)) for x in ks]
        k = 2*(a-b)/N
        s = np.full(ks.shape, 0)
        p = lagrangify(ps)
        dp = lambda x: np.array([lagGrad(i, ks, x) for i in range(len(ks))])
        for x,y in psKnown:
            dps = dp(x)
            c = y-p(x)
            s = s + c*dps
        return k*s

    # print(grad(cost)(equiX(a,b,10)))
    # print(gradient(equiX(a,b,10)))

    cp = Plottable(cost, None, None, gradient)


    bNodes = gradientDescent(cp, np.array(xs))
    pbNodes = [(x,f(x)) for x in bNodes]

    # print(cp.history)
    
    # plt.figure()
    # plt.plot(cp.history)
    # plt.show()
    
    return lagrangify(pbNodes), cp.history

class OptLagrangePol(Plottable):
    
    def __init__(self, psKnown, xStart):
        ls = [x for x,_ in psKnown]
        start, end = min(ls), max(ls)
        function, hs = optLagrangify(psKnown, xStart)
        super().__init__(function, start, end, None)
        self.history = hs

#Opt Lagrange with known function
#-----
def optLagrangifyLeg(f, psKnown, xs):

    ls = [x for x,_ in psKnown]
    a, b, N = min(ls), max(ls), len(ls)

    def cost(ks):
        ps = [(x,f(x)) for x in ks]
        k = (b-a)/N
        s = 0
        p = lagrangify(ps)
        for x,y in psKnown:
            s = s + (y-p(x))**2
        return k*s

    def lagGrad(i, ks, x):
        partOne = grad(f)(ks[i])*func.reduce(mul, [1] + [(x - ks[k])/(ks[i]-ks[k]) for k in range(len(ks)) if i != k])
        partTwo = f(ks[i])*sum([(ks[k]-x)/((ks[i]-ks[k])**2)*func.reduce(mul, [1] + [(x-ks[j])/(ks[i]-ks[j]) for j in range(len(ks)) if j != i and j != k]) for k in range(len(ks)) if k != i])
        partThree = sum([f(ks[k])*(x-ks[k])/((ks[k]-ks[i])**2)*func.reduce(mul, [1] + [(x-ks[j])/(ks[k]-ks[j]) for j in range(len(ks)) if k != j and j != i]) for k in range(len(ks)) if k != i])
        return partOne + partTwo + partThree

    def gradient(ks):
        ps = [(x,f(x)) for x in ks]
        k = 2*(a-b)/N
        s = np.full(ks.shape, 0)
        p = lagrangify(ps)
        dp = lambda x: np.array([lagGrad(i, ks, x) for i in range(len(ks))])
        for x,y in psKnown:
            dps = dp(x)
            c = y-p(x)
            s = s + c*dps
        return k*s

    cp = Plottable(cost, None, None, gradient)

    # plt.figure()
    # plt.plot(cp.history)
    # plt.show()

    bNodes = gradientDescent(cp, np.array(xs))
    pbNodes = [(x,f(x)) for x in bNodes]
    return lagrangify(pbNodes), cp.history

class OptLagrangePolLeg(Plottable):
        
    def __init__(self, f, psKnown, xStart):
        ls = [x for x,_ in psKnown]
        start, end = min(ls), max(ls)
        function, hs = optLagrangifyLeg(f, psKnown, xStart)
        super().__init__(function, start, end, None)
        self.history = hs

#Error Calculations
#-----

class ErrorCompare(Plottable):
    __slots__ = ["sqErr", "supErr", "N"]

    def __init__(self, function, mi, ma, n=10):
        super().__init__(function, mi, ma)

    @func.lru_cache(256)
    def genny(self, steps=None):
        return None

    def err2(self, n, k):
        p, f = [P[1] for P in equiNode(self.start, self.end, 100 * n, self.genny(steps=k).function)], [P[1] for P in equiNode(self.start, self.end, 100 * n, self.function)]
        return np.sqrt((self.end - self.start) / (100 * n) * sum([(y - x) ** 2 for (x, y) in zip(p, f)]))

    def errSup(self, n, k):
        p, f = [P[1] for P in equiNode(self.start, self.end, 100 * n, self.genny(steps=k).function)], [P[1] for P in equiNode(self.start, self.end, 100 * n, self.function)]
        return max([abs(y - x) for (x, y) in zip(p, f)])

    def plot(self, *args, **kwargs):
        '''Ploting the 2-norm and sup-norm as a function of the number of interpolations points used.'''
        plt.semilogy(range(1, self.N + 1), self.sqErr, *args, label="Square Error", *kwargs)
        plt.semilogy(range(1, self.N + 1), self.supErr, *args, label="Sup Error", *kwargs)
        plt.legend()

    def plot2(self, *args, **kwargs):
        plt.semilogy(range(1, self.N+1), self.sqErr, *args, label=f"Square Error with {self.nodes}nodes", *kwargs)
        plt.legend()

#Error for Lagrange Polynomials
#-----

class ErrorLagrange(ErrorCompare):
    __slots__ = ["nodes"]

    def __init__(self, function, mi, ma, n=20, nodes = "Equi"):
        super().__init__(function, mi, ma, n)
        self.nodes = nodes
        self.N = n
        self.sqErr, self.supErr = [self.err2(n, m) for m in range(1, n+1)], [self.errSup(n, m) for m in range(1, n+1)]

    @func.lru_cache(256)
    def genny(self, steps=None):
        if steps == None:
            steps = self.N
        if self.nodes == "Equi":
            return LagrangePol(equiNode(self.start, self.end, steps, self.function))
        if self.nodes == "Cheby":
            return LagrangePol(chebyNode(self.start, self.end, steps, self.function))
        raise TypeError("Not a valid nodespread")

#Error of Spline interpol
#-----

class ErrorSpline(ErrorCompare):
    __slots__ = ["K", "extras"]

    def __init__(self, function, mi, ma, n=10, k=10, extra = False):
        super().__init__(function, mi, ma)
        self.N, self.K = n, k
        if extra:
            self.extras = [[self.errSup(self.N, (k+2, np)) for k in range(self.K)] for np in range(1,11)]
        self.supErr = [self.errSup(self.N, (k+2, self.N)) for k in range(self.K)]  # Fiks dette Thomas!

    @func.lru_cache(256)
    def genny(self, steps=None):
        intervals = np.linspace(self.start, self.end, steps[0])
        intervals = [(intervals[i], intervals[i + 1]) for i in range(len(intervals) - 1)]
        pintervals = [equiNode(mi, ma, steps[1], self.function) for (mi, ma) in intervals]
        return SplinePol(pintervals)

    def extraplot(self, *args, **kwargs):
        for i in range(len(self.extras)):
            plt.semilogy([k for k in range(1, self.K+1)], self.extras[i], *args, label = f"Sup Error with n ={i+1}", **kwargs)
        plt.legend()

    def plot(self, *args, **kwargs):
        '''This is overloaded as we're plotting with another variable than the number of interpolation points'''
        # plt.semilogy(range(2, self.K+2), self.sqErr, label = "Square Error")
        plt.semilogy([self.N * i for i in range(1, self.K + 1)], self.supErr, *args, label=f"Sup Error with n={self.N}", **kwargs)
        plt.legend()

#Error of optimized Lagrange Polynomials
#-----

class ErrorOpt(ErrorCompare):
    __slots__ = ["v", "knownEqui"]

    def __init__(self, f, mi, ma, N = 1000, v = "Equi"):
        self.function, self.start, self.end, self.N = f, mi, ma, N
        self.knownEqui = equiNode(mi, ma, N, f)
        self.v = "Equi"
        self.sqErr = [self.err2(self.N, k) for k in range(1, 11)]


    def genny(self, steps = 1):
        xs = equiX(self.start, self.end, steps)
        if self.v == "Equi":
            return OptLagrangePol(self.knownEqui, xs)
        # elif self.v == "Cheby":
        #     return OptLangrangePol(self.knownCheby, steps)
        return None

    def plot(self, *args, **kwargs):
        plt.semilogy(range(1, 11), self.sqErr, *args, label="Descent - Square Error", *kwargs)
        plt.legend()

#Error of optimized Lagrange Polynomials with known function
#-----

class ErrorOptLeg(ErrorCompare):
    __slots__ = ["v", "knownEqui"]

    def __init__(self, f, mi, ma, N = 1000, v = "Equi"):
        self.function, self.start, self.end, self.N = f, mi, ma, N
        self.knownEqui = equiNode(mi, ma, N, f)
        self.v = "Equi"
        self.sqErr = [self.err2(self.N, k) for k in range(1, 11)]


    def genny(self, steps = 1):
        xs = equiX(self.start, self.end, steps)
        if self.v == "Equi":
            return OptLagrangePolLeg(self.function, self.knownEqui, xs)
        # elif self.v == "Cheby":
        #     return OptLangrangePol(self.knownCheby, steps)
        return None

    def plot(self, *args, **kwargs):
        plt.semilogy(range(1, 11), self.sqErr, *args, label="Descent - Square Error", *kwargs)
        plt.legend()
#Tests
#-----
#This is supposed to be defined on [0,1]
def a(x):
    return np.cos(2 * np.pi * x)

#This is supposed to be defined on [0,π/4]
def b(x):
    return np.exp(3 * x) * np.sin(2 * x)

def runge(x):
    return 1 / (x ** 2 + 1)

st = time.time()
p = ErrorLagrange(runge, -5, 5, n = 10)
print(f"lagrange time:{time.time()-st}")
st = time.time()
q = ErrorLagrange(runge, -5, 5, n = 10, nodes = "Cheby")
print(f"lagrange time:{time.time()-st}")
# print(p)

# st = time.time()
# q = SplinePol([equiNode(0, 0.25, 4, a), equiNode(0.25, 0.5, 4, a), equiNode(0.5, 0.75, 4, a), equiNode(0.75, 1, 4, a)])
# print(f"spline time:{time.time()-st}")
# print(q)

# st = time.time()
# p = ErrorSpline(a, 0, 1, 5, 400)
# print(f"Spline time:{time.time()-st}")
# print(p)

# st = time.time()
# r = OptLagrangePol(equiNode(-5, 5, 100, runge), equiX(-5, 5, 10))
# print(f"optimal lagrange time:{time.time()-st}")
# print(r)

st = time.time()
r  = ErrorOpt(runge, -5, 5, 1000)
print(f"Opt time:{time.time()-st}")
st = time.time()
s  = ErrorOptLeg(runge, -5, 5, 1000)
print(f"Opt time:{time.time()-st}")
plt.figure()
p.plot2('r')
q.plot2('b')
r.plot()
s.plot()
plt.show()
