import numpy as np
import GPy
from matplotlib import pyplot as plt

var = 1e-6
r = 0.4

X = np.linspace(-2, 2, 5).reshape(-1, 1)

Y = np.array([-1, -1, 1, 1, 1]).reshape(-1, 1)
print(X)

kernel = GPy.kern.Matern52(input_dim=1, variance=var, lengthscale=r)
model = GPy.models.GPRegression(X, Y, kernel, noise_var=0)

test_X = np.linspace(-2, 2, 100).reshape(-1, 1)
plot_X = np.linspace(-2, 2, 100)

mean, var = model.predict(test_X)

std = np.sqrt(var).squeeze()
mean = mean.squeeze()

plt.plot(plot_X, mean)
plt.fill_between(plot_X, (mean-std), (mean+std), color='blue', alpha=0.2)

plt.show()

