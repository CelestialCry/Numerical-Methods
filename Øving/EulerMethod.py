# Eulers method
import matplotlib.pyplot as plt
import scipy.integrate as ode
from math import sqrt

def equiX(start, end, m):
    stepSize = (end-start)/m
    return [start+i*stepSize for i in range(m+1)]

# y' = F(x, y), hvor y : R -> R, F : RxR^n -> R^n
def euler(F, start, end, steps, y0):
    h = (end-start)/steps
    xs = equiX(start, end, steps)
    ys = [y0]
    for x in xs[1:]:
        temp = []
        yn = ys[-1]
        f = F(x, yn)
        for i in range(len(yn)):
            temp.append(yn[i] + h*f[i])
        ys.append(temp)
    return ys

# y' = F(x, y), hvor y : R -> R, F : RxR^n -> R^n
def heun(F, start, end, steps, y0):
    h = (end-start)/steps
    xs = equiX(start, end, steps)
    ys = [y0]
    for x in xs[1:]:
        temp = []
        eul = []
        yn = ys[-1]
        f = F(x, yn)
        for i in range(len(yn)):
            eul.append(yn[i] + h*f[i])
        for i in range(len(yn)):
            g = F(x+h, eul)
            temp.append(yn[i] + h/2*(f[i] + g[i]))
        ys.append(temp)
    return ys

def err2(xs, ys, start, end, steps):
    return (end-start)/steps*sqrt(sum([(y-x)**2 for (x,y) in zip(xs, ys)]))

ls = equiX(0,10,1000)
α, β, δ, γ = 4, 1.2, 0.4, 1.6
F = lambda x, ys: [α*ys[0]-β*ys[0]*ys[1], δ*ys[0]*ys[1]-γ*ys[1]]
init = [10, 10]
# eTest1 = euler(F, 0, 100, 10000, init)
# eTest2 = euler(F, 0, 100, 100000, init)
# eTest3 = euler(F, 0, 100, 1000000, init)
# hTest = heun(F, 0, 100, 10000, init)
# bestTest = list(map(ode.solve_ivp(G, [0, 100], init, dense_output = True).sol, ls))

#Euler error
def EulerError(sys, ini, start, end, iterable):
    errls = []
    superIntendent = ode.solve_ivp(sys, [start, end], ini, dense_output = True)
    for i in iterable:
        ls = equiX(start, end, i)
        test = euler(sys, start, end, i, ini)
        comp = list(map(superIntendent.sol, ls))
        err = 0
        for j in range(len(test[0])):
            err += err2([t[j] for t in test], [c[j] for c in comp], start, end, i)
        errls.append(err)
    return errls

#Heun Error
def HeunError(sys, ini, start, end, iterable):
    errls = []
    superIntendent = ode.solve_ivp(sys, [start, end], ini, dense_output = True)
    for i in iterable:
        ls = equiX(start, end, i)
        test = heun(sys, start, end, i, ini)
        comp = list(map(superIntendent.sol, ls))
        err = 0
        for j in range(len(test)):
            err += err2(test[j], comp[j], start, end, i)
        errls.append(err)
    return errls

# plt.figure()
# plt.axes(xlabel = "Prey", ylabel = "Predator")
# plt.title("Lotka-Volterra System")
# # plt.plot([v[0] for v in eTest1], [v[1] for v  in eTest1], 'c')
# plt.plot([v[0] for v in eTest2], [v[1] for v  in eTest2], 'b', label = "Coarse")
# plt.plot([v[0] for v in eTest3], [v[1] for v  in eTest3], 'k', label = "Fine")
# plt.plot([v[0] for v in hTest], [v[1] for v in hTest], 'r', label = "Heun")
# plt.plot([v[0] for v in bestTest], [v[1] for v in bestTest], 'm', label = "Sci")
# plt.legend()
# plt.show()

errsEul = EulerError(F, init, 0, 10, [i*1000 for i in range(1, 101)])
errsHeun = HeunError(F, init, 0, 1, [i*1000 for i in range(1, 101)])
plotls = [1/(i*1000) for i in range(1, 101)]
errsEul.reverse()
errsHeun.reverse()
plotls.reverse()
plt.figure()
plt.axes(xlabel = "h", ylabel = "error")
plt.title(f"Convergence of Methods w/ α={α}, β={β}, δ={δ} and γ={γ}")
plt.loglog(plotls, errsEul, label = "Euler")
plt.loglog(plotls, errsHeun, label = "Heun")
plt.legend()
plt.show()