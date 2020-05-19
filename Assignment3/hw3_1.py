import numpy as np 


def Find_best_coef_LinearRegression(x, y):
    xx = np.sum(x**2)
    xy = x.dot(y)
    x_sum = np.sum(x)
    y_sum = np.sum(y)
    n = x.size
    m = ( x_sum * y_sum - n * xy ) / ( x_sum ** 2 - n * xx )
    b = ( y_sum - m * x_sum ) / n 
    print('Best Fit coefficient of linear regression')
    print(f'Best Fit coefficient m : {m}')
    print(f'Best Fit coefficient b : {b}')

def Find_best_coef_QuadraticRegression(x, y):
    row_1 = np.ones((19))
    row_2 = x.reshape((19))
    row_3 = (x**2).reshape(19)
    X = np.array([row_1, row_2, row_3]).T
    a = (np.linalg.inv(X.T @ X) @ X.T).dot(y)
    print(f'Best fit coefficient of Quadratic Regression : {a}')

if __name__ == "__main__":
    x = np.array([0.1, 0.9, 2.2, 3, 4.1, 5.2, 5.9, 6.8, 8.1, 8.7, 9.2, 11, 12.4, 14.1, 15.2, 16.8, 18.7, 19.9, 22])
    y = np.array([28.5, 27, 29, 30.5, 37.3, 36.4, 32.4, 28.5, 30, 36.1, 39, 36, 32, 28, 22, 20, 27, 40, 61])
    Find_best_coef_LinearRegression(x,y)
    print('----------------------------------')
    Find_best_coef_QuadraticRegression(x,y)


