import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
from numpy.polynomial.hermite import hermgauss
import scipy
from functools import lru_cache


def softmax_fn(x, T, phase_no):
    # Softmax function for likelihood
    """ xs.shape = [BS, n_dim]"""
    likelihood = np.exp(T * x[:, phase_no]) / np.sum(np.exp(T * x), axis=1)
    return likelihood


def like_sq(x, x0=None):
    like = softmax_fn(x, x0)
    like_sq = like ** 2
    return like_sq


@lru_cache(maxsize=2)
def herm_gauss_weigts(n, dim):
    points, weights = hermgauss(n)

    # Compute grid of points to sample from
    grid = np.meshgrid(*[points for _ in range(dim)], indexing='ij')
    grid = np.array(grid).reshape(dim, -1).T  # Reshape to [BS, n_dim]

    # Compute the product of weights for each combination
    weight_grid = np.meshgrid(*[weights for _ in range(dim)], indexing='ij')
    weight_product = np.prod(np.array(weight_grid), axis=0).reshape(-1)

    # Mask of low weight points far from origin which don't contribute much
    mask = np.linalg.norm(grid, axis=1) < np.abs(points[0])
    grid = grid[mask]
    weight_product = weight_product[mask]

    return grid, weight_product


def gauss_hermite_quadrature(mean_vector, cov_matrix, n, **softmax_kwargs):
    """
    Integrate an n-variable function with respect to an n-D Gaussian distribution using Gauss-Hermite quadrature
    with Cholesky decomposition, vectorized for performance.

    Parameters:
    func (function): The function f(x) to be integrated, where x can be an array of shape [BS, n_dim].
    n (int): The number of points to use in the quadrature for each dimension.
    mean_vector (np.array): The mean vector of the Gaussian distribution.
    cov_diagonal (np.array): The covariance matrix of the Gaussian distribution. Assume to be diagonal

    Returns:
    float: The approximate integral of the function with respect to the n-D Gaussian distribution.
    """
    dim = len(mean_vector)
    normalization_constant = 1 / ((np.pi) ** (dim / 2))

    # Apply Cholesky decomposition to the covariance matrix
    # L = la.cholesky(cov_matrix)
    L = np.sqrt(cov_matrix)

    # Prepare Gauss-Hermite quadrature points and weights for each dimension
    grid, weight_product = herm_gauss_weigts(n, dim)

    # Transform the points using the Cholesky factor and add mean
    x_transformed = mean_vector + np.sqrt(2) * grid * L

    # Evaluate the function on the entire grid of points
    func_values = softmax_fn(x_transformed, **softmax_kwargs)

    # Compute the integral as the sum of weighted function evaluations
    integral = np.sum(weight_product * func_values)

    return integral * normalization_constant


def main():
    mean = np.array([1, -1, -1])
    var_mat = np.diag([1, 0.1, 0.1])

    # Compute the integral with respect to a normal distribution
    mean_prob = gauss_hermite_quadrature(softmax_fn, 64, mean, var_mat)
    print(f'{mean_prob = }')

    # E(x^2)
    mean_sq = gauss_hermite_quadrature(like_sq, 64, mean, var_mat)
    print(f'{mean_sq = }')

    var = mean_sq - mean_prob ** 2
    print(f'{var = }')

    # Plot out the likelihood function and likelihood * normal distribution
    xs = np.linspace(-5, 5, 100)
    xs_other = np.zeros_like(xs) - 1
    xs_stack = np.stack([xs, xs_other, xs_other]).T

    ys_like = softmax_fn(xs_stack)
    normal = scipy.stats.norm.pdf(xs, mean[0], var_mat[0, 0])
    ys_prob = ys_like * normal

    plt.plot(xs, ys_like, label='likelihood')
    plt.plot(xs, ys_prob, label='probability density')
    # plt.axvline(integral_normal)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
#
