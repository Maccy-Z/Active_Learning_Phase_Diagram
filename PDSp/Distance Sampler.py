import GPy
import numpy as np
import time
import scipy
from matplotlib import pyplot as plt

from utils import model_plot, make_grid, plot_scale
from config import Config

cfg = Config()
N_phases = 3
# N_train = 9
# N_dist = 25
# N_eval = 19
# xmin, xmax, ymin, ymax = -1, 1, -2, 2

# Test phase diagram.
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


# Sample phase diagrams from models.
def gen_pd(models: list[GPy.core.GP], xs, sample=None):
    """
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
            y_pred = pd_model.posterior_samples_f(xs, full_cov=True, size=sample).squeeze(axis=1)
        y_preds.append(y_pred)

    y_preds = np.stack(y_preds)

    sample_pds = np.argmax(y_preds, axis=0).T
    return sample_pds


# Fit model to existing observations
def fit_gp() -> list[GPy.core.GP]:
    "Trains a model for each phase."
    X, _, _ = make_grid(cfg.N_train)
    true_pd = tri_pd(X)

    models = []
    for i in range(N_phases):
        phase_i = (true_pd == i) * 2 - 1  # Between -1 and 1

        kernel = GPy.kern.Matern52(input_dim=2, variance=1., lengthscale=1.)
        model = GPy.models.GPRegression(X, phase_i.reshape(-1, 1), kernel, noise_var=0.01)
        model.optimize()

        models.append(model)
    return models


def dist2(pd1s: np.ndarray, pd2s: np.ndarray):
    #n_diagrams, n_points = pd1s.shape
    diffs = np.not_equal(pd1s, pd2s)
    mean_diffs = np.mean(diffs, axis=1)

    return np.mean(mean_diffs)


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
        mu, var = model.predict(x_new.reshape(-1, 2))
        mus.append(mu.squeeze()), vars.append(var.squeeze())

    sigmas = np.sqrt(vars).tolist()
    hm_roots = scipy.special.roots_hermite(30)

    probs = []
    for i, (mu, sigma) in enumerate(zip(mus, sigmas)):
        mu_remain = mus[:i] + mus[i + 1:]
        sigma_remain = sigmas[:i] + sigmas[i + 1:]

        probs.append(max_1(mu, sigma, mu_remain, sigma_remain, hm_roots))

    return np.array(probs)


# Predict new observaion and phase diagram
def gen_pd_new_point(models: list[GPy.core.GP], x_new, sample_xs, sample):
    """Sample P_{n+1}(x_{n+1}, y), new phase diagrams assuming a new observation is taken at x_new.
    Returns: Phase diagrams"""

    # First sample P_{n}(y_{n+1})
    pd_probs = sample_new_y(models, x_new)

    # Sample new models for each possible observation
    pds = []
    for obs_phase, obs_prob in enumerate(pd_probs):
        #print(f'Prob observe phase {obs_phase}: {obs_prob:.3f}')

        new_models = []
        for phase_i, model in enumerate(models):
            X, Y = model.X, model.Y

            y_new = 2 * int(phase_i == obs_phase) - 1
            X_new, Y_new = np.vstack([X, x_new, x_new+0.01]), np.vstack([Y, y_new, y_new])

            kernel = GPy.kern.Matern52(input_dim=2, variance=1., lengthscale=1.)
            model = GPy.models.GPRegression(X_new, Y_new, kernel, noise_var=0.01)
            model.optimize()

            new_models.append(model)

        # Sample new phase diagrams, weighted to probability model is observed
        if sample is None:
            count = round(1000 * obs_prob)
            pd_new = gen_pd(new_models, sample_xs, sample=None)
            pd_new = np.repeat(pd_new, count, axis=0)
        else:
            pd_new = gen_pd(new_models, sample_xs, sample= round(sample * obs_prob)) # Note, rounding here.
        pds.append(pd_new)

    pds = np.concatenate(pds)
    return pds


def acquisition(models, new_Xs, plot=True):
    # Grid over which to compute distances
    X, _, _ = make_grid(cfg.N_dist, xmin=cfg.xmin, xmax=cfg.xmax)

    # P_n
    pd_old = gen_pd(models, X, sample=cfg.sample_old)

    # P_{n+1}
    avg_dists = []
    for new_X in new_Xs:
        print(new_X)
        pd_new = gen_pd_new_point(models, new_X, sample_xs=X, sample=cfg.sample_new)

        # Expected distance between PD1 and PD2 averaged over all pairs
        pd_old_repeat = np.repeat(pd_old, pd_new.shape[0], axis=0)
        pd_new_tile = np.tile(pd_new, (pd_old.shape[0], 1))
        avg_dist = dist2(pd_old_repeat, pd_new_tile)
        avg_dists.append(avg_dist)

    if plot:
        # Plot current P_{n} and sample of P_{n+1}
        new_point = new_Xs[1]
        avg_dist = avg_dists[1]

        X_train, _, _ = make_grid(cfg.N_train)
        xs_train, ys_train = plot_scale(X_train, cfg.N_dist,
                                        cfg.xmin, cfg.xmax, cfg.ymin, cfg.ymax)

        plt.subplot(1, 2, 1)
        plt.title("Current PD")
        plt.imshow(pd_old[0].reshape(cfg.N_dist, cfg.N_dist))
        plt.scatter(xs_train, ys_train, marker="x", s=20)
        plt.xlim([0, cfg.N_dist - 1]), plt.ylim([0, cfg.N_dist - 1])


        plt.subplot(1, 2, 2)
        plt.title(f"Sample PD, diff={avg_dist:.3g}")
        plt.imshow(pd_new[0].reshape(cfg.N_dist, cfg.N_dist))
        plt.scatter(xs_train, ys_train, marker="x", s=20)

        x_new, y_new = plot_scale(new_point.reshape(1, -1),
                                  cfg.N_dist, cfg.xmin, cfg.xmax, cfg.ymin, cfg.ymax)
        plt.scatter(x_new, y_new, c="r")
        plt.xlim([0, cfg.N_dist - 1]), plt.ylim([0, cfg.N_dist - 1])

        plt.show()

    return avg_dists


def main():

    models = fit_gp()

    new_Xs, _, _ = make_grid(cfg.N_eval, xmin=cfg.xmin, xmax=cfg.xmax)
    # new_Xs = np.array([[i, -1.25] for i in np.linspace(-1, 1, 19)])
    avg_dists = acquisition(models, new_Xs, plot=False)

    print(avg_dists)
    avg_dists = np.array(avg_dists).reshape(cfg.N_eval, cfg.N_eval)

    plt.imshow(avg_dists)

    plt.xlim([0, cfg.N_eval - 1]), plt.ylim([0, cfg.N_eval - 1])

    plt.colorbar()
    plt.show()

    # with open("./saves/both_sample", "wb") as f:
    #     pickle.dump(dists, f)




if __name__ == "__main__":
    main()


