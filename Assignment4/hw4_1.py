import numpy as np 
from scipy.optimize import minimize
from scipy.optimize import Bounds
from scipy.optimize import LinearConstraint

def objective1(x):
    '''
    objective function of problem 1
    Args:
    ---
        x (2d numpy array) : input vector
    Return:
    ---
        value : function value
    '''
    return ( 2 * x[0] * x[1] + ( x[0] ** 2 * np.pi ) / 2 ) * -1

if __name__ == "__main__":
    bounds_1 = Bounds([1, 1], [np.inf, np.inf])
    linear_constraints_1 = LinearConstraint([2+np.pi, 2], 10, 10)
    x0 = np.array([1, 1])
    res = minimize(objective1, x0, method = 'trust-constr', constraints=[linear_constraints_1], options={'disp':True}, bounds=bounds_1)
    print(f'Optimum point : {res.x}')


