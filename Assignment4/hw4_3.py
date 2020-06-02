import numpy as np 
from scipy.optimize import minimize
from scipy.optimize import linprog

def main_obj(x, r):
    '''
    combine main objective and penalty function
    Args:
    ---
        x : input vector 
        r : r value of penalty term
    Return:
    ---
        value : objective function value
    '''
    # ==== original objective and penalty function ====
    def objective3(x):
        '''
        objective of problem3-2
        Args:
        ---
            x : input vector
        Return:
        ---
            value : function value of input vector
        '''
        value = x[0] + x[1] - 50
        return -1 * value

    def g1(x):
        '''
        First penalty function
        '''
        return 50 * x[0] + 24 * x[1] - 2400
    
    def g2(x):
        '''
        Second penalty function
        '''
        return 30 * x[0] + 33 * x[1] - 2100

    def g3(x):
        '''
        Third penalty function
        '''
        return 45 - x[0]
    
    def g4(x):
        '''
        Fourth penalty function
        '''
        return 5 - x[1]
    # =================================================
    value = objective3(x) + r * ( max(0, g1(x))**2 + max(0, g2(x))**2 + max(0, g3(x))**2 + max(0, g2(x))**2 )
    return value

if __name__ == "__main__":
    print('=====Linear Programming=====')
    c = [-1,-1]
    A = [[50,24], [30,33]]
    b = [40*60,35*60]
    x0_bounds = (45, None)
    x1_bounds = (5, None)
    res3_1 = linprog(c, A_ub = A, b_ub = b, bounds=[x0_bounds, x1_bounds], method='revised simplex', options={'disp':True})
    print(f'Actual Optimum Value : {res3_1.fun * -1 - 50}')
    print(f'Optimal point (x_0, x_1) : ({res3_1.x[0]},{res3_1.x[1]})')
    print('============================')
    print('==== Exterior penalty function method ====')
    x_prev = np.array([0,0])
    r = 1
    c = np.random.randint(1,2)
    x_next = np.array([1,1])
    tol = 1e-5
    while np.linalg.norm(x_prev - x_next) > tol:
        res = minimize(main_obj, x_next, args=(r), method='nelder-mead', options={'xatol':tol, 'disp':True})
        x_prev = x_next.copy()
        x_next = res.x.copy()
        r = c * r

    print(f'Optimum point : {x_next}')
    print(f'====Bye Bye ====')

