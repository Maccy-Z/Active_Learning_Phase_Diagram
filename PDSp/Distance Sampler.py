import GPy
import numpy as np
from collections import defaultdict
import scipy

from utils import model_plot, make_grid, plot_predcitions
from matplotlib import pyplot as plt

N_phases = 3
N_train = 10
N_test = 25

def fun(x):
    x1 = x[:,0]
    x2 = x[:,1]
    return np.sign(x1**2 + x2**2 - 1)


def tri_pd(X):
    x, y = X[:, 0], X[:, 1]

    theta = np.arctan2(x, y)
    a = -4
    b = -1
    c = 1
    d = 4

    conditions = [
        (theta > a) & (theta < b),
        (theta >= b) & (theta < c),
        (theta >= c) & (theta < d)
    ]
    values = range(N_phases)
    phase_diagram = np.select(conditions, values, default=0)

    return phase_diagram


# Computes distance between two phase diagrams. Must be rectangular grid of points.
def dist(pd_1: np.ndarray, pd_2: np.ndarray):
    n_points = pd_1.shape[0]
    p_1, p_2 = pd_1.astype(int), pd_2.astype(int)

    # TODO: General non-uniform grids of points.
    phase_1, area_1 = np.unique(p_1, return_counts=True)
    phase_2, area_2 = np.unique(p_2, return_counts=True)

    # Be careful if there is a missing phase in a diagram.
    p_area_1, p_area_2  = defaultdict(int), defaultdict(int)
    p_area_1.update(zip(phase_1, area_1))
    p_area_2.update(zip(phase_2, area_2))

    # Find all present phases in either diagram
    all_phases = np.union1d(phase_1, phase_2)
    abs_diffs = []
    for phase in all_phases:
        abs_diff = np.abs(p_area_1[phase] - p_area_2[phase])
        abs_diffs.append(abs_diff)

    tot_abs_diff = np.sum(abs_diffs) / n_points / 2

    return tot_abs_diff


# Probability single gaussian is larger than the rest by integral of form f(x) exp(-x^2) where f(x) is product of CDFs.
def max_1(mu_1, sigma_1, mus, sigmas, hermite_roots):
    xs, weights = hermite_roots    # x to sample, weights for summation

    # Calculate CDF of each scaled gaussian
    cdfs = []
    for mu, sigma in zip(mus, sigmas, strict=True):
        scaled_xs = (np.sqrt(2) * sigma_1 * xs + mu_1 - mu) / sigma
        cdf_xs = scipy.stats.norm.cdf(scaled_xs)
        cdfs.append(cdf_xs)

    cdfs = np.stack(cdfs)

    # Product of cdfs for f(x)
    f = np.prod(cdfs, axis=0)

    prob = np.sum(weights * f) / np.sqrt(np.pi)

    return prob


# Probability each y is observed
def sample_new_y(models, x_new):
    """Need to find phase at x_{n+1}. Everything is gaussian, so this is finding the max of gaussians."""
    mus, vars = [], []
    for model in models:
        mu, var = model.predict(x_new)
        mus.append(mu.squeeze()), vars.append(var.squeeze())

    sigmas = np.sqrt(vars).tolist()
    hm_roots = scipy.special.roots_hermite(30)

    probs = []
    for i, (mu, sigma) in enumerate(zip(mus, sigmas)):
        mu_remain = mus[:i] + mus[i + 1:]
        sigma_remain = sigmas[:i] + sigmas[i + 1:]

        probs.append(max_1(mu, sigma, mu_remain, sigma_remain, hm_roots))

    return np.array(probs)



def gen_pd_new_point(models: list[GPy.core.GP], x_new, sample_xs, sample):
    """Sample P_{n+1}(x_{n+1}, y)
    Returns: Phase diagrams"""
    assert isinstance(sample, int) and sample != 1, "Must take a number of samples not 1"


    # First sample P_{n}(y_{n+1})
    pd_probs = sample_new_y(models, x_new)

    # Sample new models for each possible observation
    kernel = GPy.kern.Matern52(input_dim=2, variance=1., lengthscale=1.)
    pds = []
    for obs_phase, obs_prob in enumerate(pd_probs):
        print(f'Prob observe phase {obs_phase}: {obs_prob}')

        new_models = []
        for phase_i, model in enumerate(models):
            X, Y = model.X, model.Y

            y_new = int(phase_i == obs_phase)
            X_new, Y_new = np.vstack([X, x_new]), np.vstack([Y, y_new])

            model = GPy.models.GPRegression(X_new, Y_new, kernel, noise_var=0.01)
            model.optimize()

            new_models.append(model)

        # Sample new phase diagrams, weighted to probability model is observed
        pd_new = gen_pd(new_models, sample_xs, sample=round(sample * obs_prob))

        print(pd_new.shape)
        pds.append(pd_new)

    pds = np.concatenate(pds)
    return pds


def gen_pd(models: list[GPy.core.GP], xs, sample=None):
    """
    Generate phase diagram from models of each individual phase.

    @param models: List of models for each phase
    @param xs: Points to sample from
    @param sample: Use mean or sample from GP.
    @return: List of possible phase diagrams.
    """

    y_preds = []
    for phase_i, pd_model in enumerate(models):
        if sample is None:
            y_pred, _ = pd_model.predict(xs, full_cov=True)  # m.shape = [n**2, 1]
        else:
            y_pred = pd_model.posterior_samples_f(xs, full_cov=True, size=sample).squeeze()
        y_preds.append(y_pred)

    y_preds = np.stack(y_preds)

    sample_pds = np.argmax(y_preds, axis=0).T
    return sample_pds



def fit_gp() -> list[GPy.core.GP]:
    "Trains a model for each phase."
    X, _, _ = make_grid(N_train)
    true_pd = tri_pd(X)

    models = []
    for i in range(N_phases):
        phase_i = (true_pd == i) * 2 - 1  # Between -1 and 1

        kernel = GPy.kern.Matern52(input_dim=2, variance=1., lengthscale=1.)
        model = GPy.models.GPRegression(X, phase_i.reshape(-1, 1), kernel, noise_var=0.01)
        model.optimize()

        models.append(model)
    return models


def main():
    models = fit_gp()
    X, _, _ = make_grid(N_test)
    pd_old = gen_pd(models, X, sample=None)
    # pd2s = gen_pd(models, X, sample=100).T
    #
    # # Expected distance between PD1 and PD2 averaged over all instances of each other
    # dists = []
    # for pd1 in pd1s:
    #     for pd2 in pd2s:
    #         dists.append(dist(pd1, pd2))
    # avg_dist = np.mean(dists)
    # print(avg_dist)

    new_point = np.array([[0, 1]])
    pd_new = gen_pd_new_point(models, new_point, sample_xs=X, sample=5)

    plt.imshow(pd_old.reshape(25, 25))
    plt.show()
    for pd in pd_new:
        plt.imshow(pd.reshape(25, 25))
        plt.colorbar()
        plt.show()
    print(pd_new.shape)

    exit(3)
    # Train
    n_train = 30
    X, _, _ = make_grid(n_train)
    Y = fun(X).reshape(-1, 1)

    kernel = GPy.kern.RBF(input_dim=2, variance=1., lengthscale=1.)
    model = GPy.models.GPRegression(X, Y, kernel)
    model.optimize(messages=True)

    # Test
    n_test = 20
    X, _, _ = make_grid(n_test)

    ys_pred = np.sign(model.posterior_samples_f(X, full_cov=True, size=1).reshape(-1, 1))
    p1 = np.concatenate([X, ys_pred], axis=1)

    ys_pred = np.sign(model.posterior_samples_f(X, full_cov=True, size=1).reshape(-1, 1))
    p2 = np.concatenate([X, ys_pred], axis=1)

    print("Total distance:", dist(p1, p2))

    plot_predcitions(model)

    # TODO: Sample P_{n+1}:
    # Given a GP model P_n and x_{n+1}, sample y_{n+1}. Then train new GP model including datapoints {n+1}

if __name__ == "__main__":
    main()


