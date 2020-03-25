import matplotlib.pyplot as plt
import autograd.numpy as np
from autograd import jacobian, grad
from operator import mul
from functools import reduce, partial
import functools
#import _thread as thread
import time

class Point:
    """
    A class for constructing points, which are equivalent to a mutable 2-tuple.

    This class is mostly legacy code, and was used for debugging purposes.
    It is still alive as most of the code depends on this class.

    The representation of this class allows for pretty printing of points though :)

    Attributes
    ----------
    x :: Double
        x-coordinate
    y :: Double
        y-coordinate

    Operators
    ----------
    [] - Set and Get
        Let P be of class Point
        P["x"] and P["y"] can be used to get and set.
        Note: P[0] = P["x"] and P[1] = P["y"]
    print()
        It's printable, simply pu

    Raises
    ----------
    IndexError
        User have inserted an illegal index
    """
    __slots__ = ["x", "y"]

    def __init__(self, x = 0, y = 0):
        """
        Class constructor, with a default constructor which gives the origin.

        Parameters
        ----------
        x :: Double
         x-coordinate of the point
        y :: Double
            y-coordinate of the point       
        """
        self.x = x
        self.y = y


    def __repr__(self): #This is plainly a representation overload
        return "(x:" + str(self.x) + "; " + "y:" + str(self.y) + ")"

    def __setitem__(self, place, val): #Set operator
        if place == "x" or place == 0:
            self.x = val
        if place == "y" or place == 1:
            self.y = val
        else:
            raise IndexError(f"{place} is out of range")

    def __getitem__(self, place): #Get operator
        if place == "x" or place == 0:
            return self.x
        if place == "y" or place == 1:
            return self.y
        else:
            raise IndexError(f"{place} is out of range")

class Plottable():
    """
    This is a wrapper for functions, giving them some more structure.
    The nitty gritty of plotting points will be hidden inside this class.

    Attributes
    ----------
    function :: Function
        The function to wrap around
    max_dom :: Double
        Endpoint of plotting
    min_dom :: Double
        Startpoing of plotting
    
    Methods
    ----------
    plot(start = None, end = None, step = 50)
        You can input your own interval with start and end,
        and adjust the amount of points to plot with step
    diff(n)
        differentiate function n times if possible using autograd

    Operators
    ----------
    () - Call
        Can use parantheses to pass arguments to function
    print()
        Can use print to create and show plot
    """
    __slots__ = ["function", "max_dom", "min_dom"]

    def __init__(self, function = lambda x: 0, mi = 0, ma = 1):
        """
        Class constructor, with a default constructor which constructs the 0-function on [0,1].

        Parameters
        ----------
        function :: Function
            The function to wrap
        mi :: Double
            Start of interval
        ma :: Double
            End of interval
        """
        self.function, self.max_dom, self.min_dom = function, ma, mi
    
    def plot(self, *args, start = None, end = None, step = 50, **kwargs):
        """
        Plots the wrapped function within it's interval.

        Parameters
        ----------
        *args :: ???
            Arguments from plt.plot() which are not the lists to plot,
        start :: Double
            Start of interval to plot on, defaults to min_dom
        end :: Double
            End of interval to plot on, defaults to max_dom
        step :: Int
            How many points to calculate
        **kwargs :: ???
            Keywordarguments from plt.plot
        """
        if start == None or end == None:
            xs = np.linspace(self.min_dom, self.max_dom, step)
            ys = list(map(self.function, xs))
        else:
            xs = np.linspace(start, end, step)
            ys = list(map(self.function, xs))
        plt.plot(xs, ys, *args, *kwargs)
        plt.legend()
    
    def __repr__(self): #Class representation
        plt.figure()
        self.plot()
        plt.show()

    def __call__(self, *args): #Function calling overloading
        return self.function(args)
    
    def diff(self, n):
        """
        Differentiation method using jacobian from autograd

        Parameters
        ----------
        n :: Int
            Number of times to differentiate the wrapped function

        Returns
        ----------
        Returns the differentiated function
        """
        n = self.function
        for i in range(n):
            n = jacobian(n)
        return n

class Lagrange(Plottable):
    """
    A wrapper for the Lagrange Polynomial of a given set of points.
    The polynomial is generated at construction.

    Super-class
    ----------
    Plottable - Important changes
        Attributes
        ----------
        function :: Function
            This is the Lagrange polynomial

    Attributes
    ----------
    points :: [Point]
        A list of objects of the class Point

    Methods
    ----------
    sep()
        Separates the points in the list and returns the new list
    """
    __slots__ = ["points"]

    #Strangely enough the way this Lagrange Polynomial is defined is well defined within autograd!
    #How do I comment on this thingy mac-jiggy???
    def __init__(self, plist):
        """
        Class constructor, not equipped with a default constructor.

        Parameters
        ----------
        plist :: [Point]
            A list of points to do the interpolation over
        """
        self.points = plist
        xs, ys = self.sep()
        self.max_dom, self.min_dom = max(xs), min(xs)
        # Everything above should make sense, everything below is a clusterfuck
        λj = lambda j, ls, x: ys[j]*reduce(mul, [(x-arg)/(ls[j]-arg) for arg in ls if ls[j] != arg])
        self.function = lambda x: sum([λj(i, xs, x) for i in range(len(xs))])

    def sep(self):
        """
        Separates the points in the list.

        Returns
        ----------
        Returns two list, one consisting of x-coordinates, the other of y-coordinates
        """
        return [p["x"] for p in self.points], [p["y"] for p in self.points]

class PiecewiseLagrange(Lagrange):
    """
    A wrapper for lagrange functions defined on several intervals piecewise.

    Super-Class
    ----------
    Plottable -> Lagrange - Important changes
        Attributes
        ----------
        function :: Function
            This is the piecewise Lagrange Polynomial
    
    Attributes
    ----------
    functions :: [Function]
        A list of Lagrange Polynomials
    intervals :: [(Double, Double)]
        A list of min_dom and max_dom for every Lagrange Polynomial, respectively
    NB: There is a 1-1 correspondance between functions and intervals defined in the canonical way

    Methods
    ----------
    sep() - overloaded
        This returns a list of separations for every Lagrange Polynomial
    nfunction(x)
        This is simply the "direct sum" of each Lagrage Polynomial on it's interval.
        It is extended to -inf and inf on the endpoints
        
    """
    __slots__ = ["functions", "interval"]

    def __init__(self, pinterval):
        """
        Constructor for this class, has no default constructor.
        This function just does the Lagrange interpolation on the given subintervals.

        Parameters
        ----------
        pinterval :: [[Point]]
            A list of lists of objects from the class Point
        """
        self.points = pinterval
        self.functions = [Lagrange(plist) for plist in pinterval] # This is where we do the lagrange
        self.interval = [(lambda x: (min(x), max(x)))(l.sep()[0]) for l in self.functions] #Defining the start and end of each Polynomial
        self.function = lambda x: self.nfunction(x) #"Gluing" the polynomials together, extending the ends
        self.min_dom, self.max_dom = self.interval[0][0], self.interval[-1][1] #Then setting the total start and endpoint to be the start of the first and end of the last function

    def sep(self):
        """
        An overload of the seperation method.

        Returns
        ----------
        Returns a list of tuples of seperations for each given interval
        """
        return [p.sep() for p in functions]

    def nfunction(self, x):
        """
        The piecewise Lagrange Polynomial

        Parameters
        ----------
        x :: Double??? #It seems Python has no well-definedness for function types
            This argument is determined by the functions input

        Returns
        ----------
        Returns the value of the piecewise Lagrange Polynomial evaluated at x
        """
        if x < self.interval[0][1]:
            return self.functions[0](x)
        for i in range(1, len(self.interval)-1):
            (m, n) = self.interval[i]
            if m < x and x < n:
                return self.functions[i](x)
        return self.functions[-1](x)


def equiNode(start, end, step, f = (lambda x: 0)):
    """
    Creation of equidistant nodes with respect to distances of x's

    Parameters
    ----------
    start :: Double
        start of interval
    end :: Double
        end of interval
    step :: Double
        number of steps
    f :: Function
        A function to distribute the y-values on

    Returns
    ----------
    A list of Point objects
    """
    xs = np.linspace(start, end, step)
    ys = map(f, xs)
    return [Point(x,y) for (x,y) in zip(xs,ys)]

def chebyNode(start, end, steps, f = lambda x: 0):
    """
    Creation of Chebyshev nodes with respect to the distance of the x's

    Parameters
    ----------
    start :: Double
        start of interval
    end :: Double
        end of interval
    step :: Double
        size of each step
    f :: Function
        A function to distribute the y-values on

    Returns
    ----------
    A list of Point objects
    """
    xs = [(end-start)/2*(np.cos(np.pi*(2*x+1)/(2*steps)))+(end+start)/2 for x in range(steps)]
    ys = map(f, xs)
    return [Point(x,y) for (x,y) in zip(xs, ys)]

def runge(x):
    """
    The runge function

    Parameters
    ----------
    x :: Double
        The value to evaluate in
    Returns
    ----------
    Returns the evaluation at x
    """
    return 1/(x**2+1)

class ErrorCompare(Plottable):
    """
    This class will act as the base for the comparisons method for the errors of the Lagrange polynomials.
    This is meant to be a generic, virtual class, so some of the functions might be missing, but defined.

    Super-Class
    ----------
    Plottable - Important changes
        Attributes
        ----------
            function :: Function
                This is the true function we want to approximate

    Attributes
    ----------
    sqErr :: [Double]
        List of error-value of the 2-norm
    supErr :: [Double]
        List of error-values of the sup-norm
    N :: Int
        maximal number of iterations

    Methods
    ----------
    genny(steps = None)
        A virtual method to generate different approximations
    err2(n)
        A method to find 2-norm error based on genny()
    errSup(n)
        A method to find sup-nomr error based on genny()
    plot() - changed
        plot now plots both 2Err and supErr
    """
    __slots__ = ["sqErr", "supErr", "N"]

    def __init__(self, function, mi, ma, n = 10):
        """
        Class Constructor, has no default constructor.

        Parameters
        ----------
        function :: Function
            The true function to approximate
        mi :: Double
            start of test domain
        ma :: Double
            end of test domain
        n :: Int
            number of iterations to run
        """
        super().__init__(function, mi, ma)

    @functools.lru_cache
    def genny(self, steps = None):
        """
        This is simply a virtual method to generate approximations.

        Parameters
        ----------
        steps :: Int
            The amount of steps to take in the approx

        Returns
        ----------
        This function returns None as it is virtual
        """
        return None


    def err2(self, n, k):
        """
        A method to find the 2-norm error of the approximation on the test interval.

        Parameters
        ----------
        n :: Int
            The amount of nodes to interpolate on
        k :: Int
            virtual variable - aka. genny() magic

        Returns
        ----------
        Returns the 2-norm error
        """
        p, f = [P["y"] for P in equiNode(self.min_dom, self.max_dom, 100*n, self.genny(steps = k).function)], [P["y"] for P in equiNode(self.min_dom, self.max_dom, 100*n, self.function)]
        return np.sqrt((self.max_dom-self.min_dom)/(100*n)*sum([(y-x)**2 for (x,y) in zip(p,f)]))

    def errSup(self, n, k):
        """
        A method to find the sup-norm error of the approximation on the test interval.
       
        Parameters
        ----------
        n :: Int
            The amount of nodes to interpolate on
        k :: Int
            virtual variable - aka. genny() magic
       
        Returns
        ----------
        Returns the sup-norm error
        """
        p, f = [P["y"] for P in equiNode(self.min_dom, self.max_dom, 100*n, self.genny(steps = k).function)], [P["y"] for P in equiNode(self.min_dom, self.max_dom, 100*n, self.function)]
        return max([abs(y-x) for (x,y) in zip(p,f)])
    
    def plot(self, *args, **kwargs):
        '''Ploting the 2-norm and sup-norm as a function of the number of interpolations points used.'''
        plt.semilogy(range(2, self.N+1), self.sqErr, *args, label = "Square Error", *kwargs)
        plt.semilogy(range(2, self.N+1), self.supErr, *args, label = "Sup Error", *kwargs)
        plt.legend()


class ErrorLagrange(ErrorCompare):
    """
    Class for comparing the Lagrange Polynomial to the original function.

    Super-Class
    ----------
    Plottable -> ErrorCompare

    Attributes
    ----------
    v :: String
        The type of node-spread (Equidistant or Chebyshev)

    Methods
    ----------
    genny(steps = None) - overloaded
        A generator function for finding a fitting Lagrange Polynomial based on the string v

    """
    __slots__ = ["v"]

    def __init__(self, function, mi, ma, n = 10, v = "Equi"):
        """
        Class constructor, has not a default constructor.

        Parameters
        ----------
        function :: Function
            The function to approximate with Lagrange Interpolation
        mi :: Double
            start of test interval
        ma :: Double
            end of test interval
        n :: Int
            number of steps to take
        v :: String
            The type of node-spread to use. Can have "Equi" or "Cheby" as input, else it will raise an execption
        """
        super().__init__(function, mi, ma, n)
        self.v = v
        self.N = n
        self.sqErr, self.supErr = [self.err2(self.N, m+1) for m in range(1, n)], [self.errSup(self.N, m+1) for m in range(1, n)]

    @functools.lru_cache
    def genny(self, steps = None):
        """
        The generator function for the Lagrange Polynomial.

        Parameters
        ----------
        steps :: Int
            number of steps to take

        Returns
        ----------
        Returns the Lagrange Polynomial with degree steps-1
        """
        if steps == None:
            steps = self.N
        if self.v == "Equi":
            return Lagrange(equiNode(self.min_dom, self.max_dom, steps, self.function))
        if self.v == "Cheby":
            return Lagrange(chebyNode(self.min_dom, self.max_dom, steps, self.function))
        raise TypeError("Not a valid node-spread")

class ErrorPiecewiseLagrange(ErrorCompare):
    """
    Class for comparing the piecwise Lagrange Polynomial to the original function.

    Super-Class
    ----------
    Plottable -> ErrorCompare

    Attributes
    ----------
    K :: Int
        The maximum amount of intervals to interpolate on

    Methods
    ----------
    genny(steps = None) - overloaded
        A generator function for finding a fitting piecewise Lagrange Polynomial based on the amount of intervals
    plot() - overloaded
        plotting is overloaded to handle intervals rather than interpolation points
    """
    __slots__ = ["K"]

    def __init__(self, function, mi, ma, n = 10, k = 10):
        """
        Class constructor, has no defualt constructor

        Parameters
        ----------
        function :: Function
            The true function to approximate
        mi :: Double
            start of test interval
        ma :: Double
            end of test interval
        n :: Int
            the exact amount of interpolation nodes in each interval
        k :: the maximum amount of intervals
        """
        super().__init__(function, mi, ma)
        self.N, self.K = n, k
        #self.sqErr = [self.err2(self.N, k+2) for k in range(self.K)]
        self.supErr = [self.errSup(self.N, k+2) for k in range(self.K)] 

    @functools.lru_cache
    def genny(self, steps = None):
        """
        The generator function for the piecewise Lagrange Polynomial with equidistant nodes.

        Parameter
        ----------
        k :: Int
            The amount of intervals to use

        Returns
        ----------
        Returns the piecewise Lagrange Polynomial
        """
        intervals = np.linspace(self.min_dom, self.max_dom, steps)
        intervals = [(intervals[i], intervals[i+1]) for i in range(len(intervals)-1)]
        pintervals = [equiNode(mi, ma, self.N, self.function) for (mi, ma) in intervals]
        return PiecewiseLagrange(pintervals)

    def plot(self, *args, **kwargs):
        '''This is overloaded as we're plotting with another variable than the number of interpolation points'''
        #plt.semilogy(range(2, self.K+2), self.sqErr, label = "Square Error")
        plt.semilogy(range(2, self.K+2), self.supErr, *args, label = "Sup Error", *kwargs)
        plt.legend()


# This is supposed to be defined on [0,1]
def a(x):
    return np.cos(2*np.pi*x)

# This is supposed to be defined on [0,π/4]
def b(x):
    return np.exp(3*x)*np.sin(2*x)


# Task i)
start = time.time()
plt.figure()
r = Lagrange(chebyNode(-5, 5, 30, runge))
r.plot()
p = Lagrange(equiNode(-5, 5, 10, runge))
p.plot()
# plt.legend()
# stop = time.time()
# plt.show()
# print(f"Time taken: {stop - start}")

# Task ii)
# start = time.time()
plt.figure()
# u = ErrorLagrange(a, 0, 1, n = 50)
# u.plot()
v = ErrorLagrange(a, 0, 1, n = 20, v = "Equi")
v.plot()
# plt.show()


plt.figure()
u = ErrorLagrange(b, 0, np.pi/4, 20)
u.plot()
# plt.show()



# Task iii)
plt.figure()
u = ErrorPiecewiseLagrange(a, 0, 1, 2, 20)
u.plot()
stop = time.time()
plt.show()
print(f"Time taken: {stop-start}")


"""
#Sammenlign
ps = chebyNode(-1, 1, 100, lambda x: 20*np.cos(np.pi*x))
s = Lagrange(ps)
plt.figure()
s.plot(400)
plt.show()
"""