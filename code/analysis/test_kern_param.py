import numpy as np
import GPy
from matplotlib import pyplot as plt
from scipy import stats

var = 10
r = 2
noise = 1e-3

X = np.linspace(-2, 1, 10).reshape(-1, 1)
X = np.array([-2, -1, 0, 1, 1.4, 1.3, 1.5, 4]).reshape(-1, 1)
Y = 2 * (X > 1).astype(int) - 1

kernel = GPy.kern.Matern52(input_dim=1, variance=var, lengthscale=r)
model = GPy.models.GPRegression(X, Y, kernel, noise_var=noise)

# Second opposite model
Y2 = -Y
kernel = GPy.kern.Matern52(input_dim=1, variance=var, lengthscale=r)
model2 = GPy.models.GPRegression(X, Y2, kernel, noise_var=noise)


test_X = np.linspace(-2, 4, 100).reshape(-1, 1)
plot_X = np.linspace(-2, 4, 100)

mean1, var1 = model.predict(test_X)
mean2, var2 = model2.predict(test_X)

std1 = np.sqrt(var1).squeeze()
mean1 = mean1.squeeze()

std2 = np.sqrt(var2).squeeze()
mean2 = mean2.squeeze()

# Plot predictions

plt.subplot(1, 2, 1)
plt.plot(plot_X, mean1)
plt.fill_between(plot_X, (mean1-std1), (mean1+std1), color='blue', alpha=0.2)
#plt.plot(plot_X, mean2)
#plt.fill_between(plot_X, (mean2-std2), (mean2+std2), color='orange', alpha=0.2)
plt.scatter(X, Y)
#plt.scatter(X, Y2)

# Probability that Y1 > Y2, and entropy to plot
mu1 = mean1# mean of first Gaussian
sigma1 = std1# standard deviation of first Gaussian
mu2 = mean2 # mean of second Gaussian
sigma2 = std2 # standard deviation of second Gaussian

mu_diff = mu1 - mu2
sigma_diff = np.sqrt(sigma1**2 + sigma2**2)   # Pythagorean addition of variances

p = stats.norm.cdf(0, loc=mu_diff, scale=sigma_diff)  # probability that X > Y
q = 1-p

H = 4 * p * q

plt.subplot(1, 2, 2)
plt.plot(plot_X, H)


print(p.shape)

plt.show()

