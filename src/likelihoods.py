import matplotlib.pyplot as plt
import numpy as np
import time
from numpy.polynomial.hermite import hermgauss
import scipy
from functools import lru_cache
from numba import jit


@jit(nopython=True)
def row_max(mat):
    max_vals = np.empty(mat.shape[0])
    for i in range(mat.shape[0]):
        max_vals[i] = np.max(mat[i])
    return max_vals


@jit(nopython=True, fastmath=True)
def softmax_fn(logits):
    """ logits.shape = [BS, n_dim]"""

    max_vals = row_max(logits)
    for i in range(logits.shape[0]):
        logits[i] -= max_vals[i]

    exp_logits = np.exp(logits)
    sum_exp = np.empty(exp_logits.shape[0])
    for i in range(exp_logits.shape[0]):
        sum_exp[i] = np.sum(exp_logits[i])

    for i in range(exp_logits.shape[0]):
        exp_logits[i] /= sum_exp[i]

    return exp_logits


@lru_cache(maxsize=5)
def herm_gauss_weigts(n, dim) -> (np.ndarray, np.ndarray):
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

    # Normalization constant
    normalization_constant = 1 / ((np.pi) ** (dim / 2))
    weight_product *= normalization_constant

    # sqrt2 grid for later use
    grid = np.sqrt(2) * grid
    return grid, weight_product


def gauss_hermite_quadrature(mean_vector, cov_matrix, n):
    """
    Integrate an n-variable function with respect to an n-D Gaussian distribution using Gauss-Hermite quadrature
    with Cholesky decomposition, vectorized for performance, with last integral skipped.

    Parameters:
    func (function): The function f(x) to be integrated, where x can be an array of shape [BS, n_dim].
    n (int): The number of points to use in the quadrature for each dimension.
    mean_vector (np.array): The mean vector of the Gaussian distribution.
    cov_diagonal (np.array): The covariance matrix of the Gaussian distribution. Assume to be diagonal

    Returns:
    float: The approximate integral of the function with respect to the n-D Gaussian distribution. Shape = [n_dim]
    """

    # Prepare Gauss-Hermite quadrature points and weights for each dimension
    dim = len(mean_vector)
    sqrt2_grid, weight_product = herm_gauss_weigts(n, dim)

    probs = gauss_hermite_quadrature_inner(mean_vector, cov_matrix, sqrt2_grid, weight_product)
    return probs


@jit(nopython=True, fastmath=True)
def gauss_hermite_quadrature_inner(mean_vector, cov_diag, sqrt2_grid, weight_product):
    # Apply Cholesky decomposition to the covariance matrix
    L = np.sqrt(cov_diag)

    # Transform the points using the Cholesky factor and add mean
    x_transformed = mean_vector + sqrt2_grid * L

    # Evaluate the function on the entire grid of points
    func_values = softmax_fn(x_transformed)

    # Only compute n-1 integrals, last one is known from summing probs.
    func_values = func_values[:, :-1]
    # Compute the integral as the sum of weighted function evaluations
    probs = np.sum(weight_product[:, np.newaxis] * func_values, axis=0)

    # Fill in last value
    remain_prob = 1 - np.sum(probs)
    probs = np.append(probs, remain_prob)

    return probs


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
