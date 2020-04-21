# Rombergs algoritme
from numpy import exp
from copy import deepcopy

def equiX(start, end, m):
    stepSize = (end-start)/m
    return [start+i*stepSize for i in range(m+1)]


def compTrap(f, start, end, m):
    h = (end-start)/m
    xs = equiX(start, end, m)
    midPoints = sum([f(xs[i]) for i in range(1, m)])
    return h*(f(xs[0])/2+midPoints+f(xs[-1])/2)

def romberg(f, start, end, maximum):
    zeros = [0 for i in range(maximum)]
    initial = [compTrap(f, start, end, 2**m) for m in range(1,maximum+1)]
    algoMatrix = [initial if i == 0 else deepcopy(zeros) for i in range(maximum)]
    for k in range(1, maximum):
        for i in range(k, maximum):
            algoMatrix[k][i] = ((4**k)*algoMatrix[k-1][i]-algoMatrix[k-1][i-1])/((4**k)-1)
    return algoMatrix


def f(x):
    return exp(-1*(x**2))
print(romberg(f, 0, 1, 4)[2][3])