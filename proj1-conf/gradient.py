import numpy as np

def grad(x):
    x1, x2 = x
    grad1 = 2.0 * x1 + 3.0 * x2 + 2.0
    grad2 = 3.0 * x1 + 18.0 * x2 - 5.0
    return np.array([grad1, grad2])

def gradient_Dual(x):
    x1, x2 = x
    grad1 = 3 - 2 * x1 - x2
    grad2 = 3 - x1 - 2 * x2
    return np.array([grad1,grad2])
