import numpy as np
def derivative(f, x, h = 1e-14):
    return (f(x+h)-f(x))/h

def meanValDiff(f, a, b):
    """
    One dimensional differentiation based on the mean value theorem
    """
    return (f(b)-f(a))/np.linalg.norm(b-a)

def meanValGrad(f, x, h = 1e-14):
    # return [derivative]
    return [meanValDiff(f, np.array([x[j]-h if i == j else x[j] for j in range(len(x))]), np.array([x[j]+h if i == j else x[j] for j in range(len(x))])) for i in range(len(x))]

a = lambda x: x[0]**3+x[1]**2
print(meanValGrad(a, np.array([2,1])))

with open("tests.txt","w") as file:
    file.write("hello world!")
    file.write("bye felicia")
    pass