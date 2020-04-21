#Second order boundary value problmer solver
import numpy as np
import matplotlib.pyplot as plt

def equiX(start, end, m):
    stepSize = (end-start)/m
    return np.array([start+i*stepSize for i in range(m+1)])

def Ah(dim):
    one = np.array([1 for i in range(dim-1)])
    two = np.array([-2 for i in range(dim)])
    return np.diag(two) + np.diag(one, -1) + np.diag(one, 1)


def bvp(ω, F, start, end, α, β, M):
    h = (end-start)/(M+1)
    def choice(i):
        if i == 1:
            return α/(h**2)
        elif i == M:
            return β/(h**2)
        else:
            return 0

    xs = equiX(start, end, M+1)
    Gh = (1/h**2)*Ah(M) + ω**2*np.identity(M)
    print(Gh)
    b = np.array([F[i-1](xs[i]) + choice(i) for i in range(1,M+1)])
    return np.linalg.solve(Gh, b)

plt.figure()
plt.plot([i/1000 for i in range(1, 1001)], bvp(1,[lambda x: 0 for i in range(1000)], 0, 1, 0, 1, 1000))
plt.show()