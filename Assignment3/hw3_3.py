import numpy as np 
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

X = np.array([0.1, 0.9, 2.2, 3, 4.1, 5.2, 5.9, 6.8, 8.1, 8.7, 9.2, 11, 12.4, 14.1, 15.2, 16.8, 18.7, 19.9, 22])
Y = np.array([28.5, 27, 29, 30.5, 37.3, 36.4, 32.4, 28.5, 30, 36.1, 39, 36, 32, 28, 22, 20, 27, 40, 61])

X = X[:, np.newaxis]
Y = Y[:, np.newaxis]

print('Try polynomial regression at degree 6, 8 and 10')
colors = ['teal', 'yellowgreen', 'gold']
lw = 2
plt.plot(X, Y, color = 'cornflowerblue', linewidth = lw, label = 'ground truth')
plt.scatter(X, Y, color = 'navy', s = 30, marker='o', label = 'training points')
for count, degree in enumerate([6,8,10]):
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model.fit(X, Y)
    y_plot = model.predict(X)
    plt.plot(X, y_plot, color = colors[count], linewidth=lw, label = 'degree %d' % degree)
plt.legend(loc='lower left')
plt.show()

