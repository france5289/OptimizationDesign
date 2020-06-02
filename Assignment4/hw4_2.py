import numpy as np 
from scipy.optimize import minimize
from scipy.optimize import Bounds
from scipy.optimize import LinearConstraint

def objective_2(x):
    '''
    objective function of problem2
    Args:
    ---
        x : input vector
    Return:
    ---
        value : function value at vector x
    '''
    A = np.array([11.5, 92.5, 44.3, 98.1, 20.1, 6.1, 45.5, 31.0, 44.3])
    value = np.sum(np.divide(x, A)**2)
    return value 

if __name__ == "__main__":
    bounds_F = Bounds([0,0,0,0,0,0,0,0,0], [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf])
    d = np.array([0.0298, 0.044, 0.044, 0.0138, 0.0329, 0.0329, 0.0279, 0.025, 0.025, 0.0619, 0.0317, 0.0368])
    linear_constraints_D = LinearConstraint([[d[0], -1*d[1], -1*d[2], 0, 0, 0, 0, 0, 0], 
                                            [0, 0, -1*d[3], d[4], d[5], -1*d[7], -1*d[8], 0, 0],
                                            [0, 0, 0, 0, d[6], 0, -1*d[9], d[-2], -1*d[-1]]
                                            ], [4, 33, 31], [4, 33, 31])
    np.random.seed(1024)
    F1 = np.ones(9)
    print(f'Initial vector : {F1}')
    prob2_res = minimize(objective_2, F1, method='trust-constr', constraints=[linear_constraints_D], options={'disp':True}, bounds=bounds_F)
    print(f'Optimum Point : {prob2_res.x}')
    print(f'Optimal Value : {objective_2(prob2_res.x)}')