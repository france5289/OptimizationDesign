# Optimization Homework 4 Report
## Note 
若 markdown 無法顯示可以點以下連結:  
- [hw4 Report](https://hackmd.io/rk4FYlAQQamzP_FaY_7JLw)
## 程式開發環境
- 使用語言
  - `Python3.7.5`
- 所需套件
  - `numpy` 
  - `scipy`
## Problem 1
### Objective function
#### Notation
- $x_1$ : 半徑
- $x_2$ : 長方形長

$$
\text{Maximize : }F(x_1 , x_2) = 2x_{1}x_{2} + (2\pi x_{1}) \\
\text{s.t : } 2x_{1}+2x_{2}+x_{1}*\pi = 10 \\
x_{1} \gt 0\\
x_{2} \gt 0\\
$$
### Solution
利用 `Scipy` 中的 `optimize` 模組直接計算 constrained minimization，使用的演算法為 `Trust-Region Constrained Algorithm`。  
#### Define objective function
```python
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
```
#### Define bounds of $x_1$ and $x_2$
```python
from scipy.optimize import Bounds
bounds_1 = Bounds([1, 1], [np.inf, np.inf])
```
#### Define Linear Constraints
```python
from scipy.optimize import LinearConstraint
linear_constraints_1 = LinearConstraint([2+np.pi, 2], 10, 10)
```
#### Solve optimization problem
```python
x0 = np.array([1, 1])
res = minimize(objective1, x0, method = 'trust-constr', constraints=[linear_constraints_1], options={'disp':True}, bounds=bounds_1)
```
output : 
```bash
`gtol` termination condition is satisfied.
Number of iterations: 12, function evaluations: 21, CG iterations: 6, optimality: 8.15e-09, constraint violation: 0.00e+00, execution time: 0.024 s.
```
#### Print optimum point
```python
print(res.x)
```
```bash
[1.40023028 1.40029313]
```
## Problem 2 
同樣利用 `Scipy` 中的 `optimize` 模組直接做 constrained minimization，使用的演算法為`Trust-Region Constrained Algorithm`
### Objective function
$$
\text{minimize } Z = \sum_{i=1}^{i=9}(\frac{F_i}{A_i})^2 \\
\text{s.t } f_1 = d_{1}F_{1}-d_{2}F_{2}-d_{3a}F_{3}-M_1 = 0 \\
f_2 = -d_{3k}F_{3}+d_{4}F_{4}+d_{5k}F_{5}-d_{6}F_{6}-d_{7k}F_{7}-M_{2} = 0 \\
f_3 = d_{5h}F_{5}-d_{7h}F_{7}+d_{8}F_{8}-d_{9}F_{9}-M_3 = 0 \\
F_{i} \ge 0 \ (i = 1, 2, \dots ,9) \\
$$
#### Notation
$$
M_1 = 4 \\
M_2 = 33 \\
M_3 = 31 \\ 
\vec{d} = \begin{bmatrix}
    0.0298 \\
    0.044 \\
    0.044 \\
    0.0138 \\
    0.0329 \\
    0.0329 \\
    0.0279 \\
    0.025 \\
    0.0619 \\
    0.0317 \\
    0.0368 
\end{bmatrix}\\
\vec{A} = \begin{bmatrix}
    11.5 \\
    92.5 \\
    44.3 \\
    98.1 \\
    20.1 \\
    6.1 \\
    45.5 \\
    31.0 \\
    44.3
\end{bmatrix}
$$
### Solution
#### Define objective function
```python
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
```
#### Define bounds of $F_i$
```python
bounds_F = Bounds([0,0,0,0,0,0,0,0,0], [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf])
```
#### Define Linear Constraints
```python
d = np.array([0.0298, 0.044, 0.044, 0.0138, 0.0329, 0.0329, 0.0279, 0.025, 0.025, 0.0619, 0.0317, 0.0368])
linear_constraints_D = LinearConstraint([[d[0], -1*d[1], -1*d[2], 0, 0, 0, 0, 0, 0], 
                                         [0, 0, -1*d[3], d[4], d[5], -1*d[7], -1*d[8], 0, 0],
                                         [0, 0, 0, 0, d[6], 0, -1*d[9], d[-2], -1*d[-1]]
                                        ], [4, 33, 31], [4, 33, 31])
```
#### Solve Optimization problem
```python
np.random.seed(1024)
F1 = np.ones(9)
print(F1)
prob2_res = minimize(objective_2, F1, method='trust-constr', constraints=[linear_constraints_D], options={'disp':True}, bounds=bounds_F)
```
**output**
```bash
[1. 1. 1. 1. 1. 1. 1. 1. 1.]
`gtol` termination condition is satisfied.
Number of iterations: 279, function evaluations: 2710, CG iterations: 474, optimality: 4.51e-09, constraint violation: 7.11e-15, execution time: 0.77 s.
```
**optimal point**
```python
print(prob2_res.x)
```
```bash
[1.34228188e+02 1.57343175e-07 2.08046685e-08 7.07677227e+02
 2.95362289e+02 3.24053147e-06 6.16636711e-08 7.17961897e+02
 2.47025848e-07]
```
**Check optimal value**
Paper 中的 optimal value 為 `940.6`
```python
print(objective_2(prob2_res.x))
```
```bash
940.5962998908748
```
## Problem 3
### Objective function
$$
\text{Maximize : }f(x_1, x_2) = x_1 + x_2 - 50 \\
\text{s.t : }50x_1 + 24x_2 \le 2400 \\
30x_1 + 33x_2 \le 2100 \\
x_1 \ge 45 \\
x_2 \ge 5
$$
### Problem 3-1
利用 `Scipy` 中 `optimize`模組的 `linprog` function 執行 linear programming 計算找出最佳解
#### Define coefficient , bounds and constraints
```python
from scipy.optimize import linprog
c = [-1,-1]
A = [[50,24], [30,33]]
b = [40*60,35*60]
x0_bounds = (45, None)
x1_bounds = (5, None)
```
#### Solve optimization problem
```python
res3_1 = linprog(c, A_ub = A, b_ub = b, bounds=[x0_bounds, x1_bounds], method='revised simplex')
print(res3_1)
```
**output**
```bash
con: array([], dtype=float64)
     fun: -51.25
 message: 'Optimization terminated successfully.'
     nit: 2
   slack: array([  0.  , 543.75])
  status: 0
 success: True
       x: array([45.  ,  6.25])
```
#### Check optimium point and value
```python
print(f'Actual Optimum Value : {res3_1.fun * -1 - 50}')
print(f'Optimal point (x_0, x_1) : ({res3_1.x[0]},{res3_1.x[1]})')
```
**output**
```bash
Actual Optimum Value : 1.25
Optimal point (x_0, x_1) : (45.0,6.25)
```
輸出與答案相同，故 optimization 結果正確
### Probelm 3-2
利用 exterior method 將 objective function 與 constrained 合併，而因為要最大化 $x_1+x_2-50$ 但我們要用 最小化演算法找出最佳解，因此將原本的 objective function 轉為 $-x_1-x_2+50$
$$
\rho(r_{k}, \vec{x})=f(\vec{x}) + r_{k}\sum_{j=1}^{m}(\max(0,g_{j}(\vec{x}))^2) \\
f(x_1, x_2) = -x_1 + -x_2 + 50\\
g_1(x_1, x_2) = 50x_1 + 24x_2 - 2400\\
g_2(x_1, x_2) = 30x_1 + 32x_2 - 2100\\
g_3(x_1) = 45-x_1\\
g(4)(x_2) = 5-x_2
$$
#### Define objective function
```python
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
```
#### Implement algorithm
依照講義上的演算法流程實作演算法，而 unconstrained optimization algorithm 則使用 `Nelder-Mead`  
```python
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
```
**output**
```bash
Optimum point : [44.45833227  7.37934256]
```
發現出來的最佳點違反了 constrained，即使我將停止條件設的更嚴格還是一樣。