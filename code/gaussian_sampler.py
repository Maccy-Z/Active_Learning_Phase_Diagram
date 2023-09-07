#import edit_source_files
import GPy
import numpy as np
import time
import scipy

from utils import ObsHolder, make_grid, to_real_scale
from config import Config
from matplotlib import pyplot as plt

np.set_printoptions(precision=2)


# Sample phase diagrams from models.
def gen_pd(models: list[GPy.core.GP], xs, cfg, sample=None):
    """
    @param models: List of models for each phase
    @param xs: Points to sample from
    @param sample: Use mean or sample from GP.
    @return: List of possible phase diagrams.
    """
    y_preds = []
    for phase_i, pd_model in enumerate(models):
        if sample is None:
            y_pred, _ = pd_model.predict(xs, include_likelihood=False)  # m.shape = [n**2, 1]
        else:
            y_pred = pd_model.posterior_samples_f(xs, size=sample, method=cfg.normal_sample).squeeze(axis=1)
        y_preds.append(y_pred)

    y_preds = np.stack(y_preds)
    sample_pds = np.argmax(y_preds, axis=0).T
    return sample_pds


# Fit model to existing observations
def fit_gp(obs_holder: ObsHolder, cfg) -> list[GPy.core.GP]:
    "Trains a model for each phase."
    X, Y = obs_holder.get_obs()

    var, r = obs_holder.get_kern_param()

    models = []
    for i in range(cfg.N_phases):
        phase_i = (Y == i)  # * 2 - 1  # Between 0 and 1

        kernel = GPy.kern.Matern52(input_dim=2, variance=var, lengthscale=r)
        # model = GPy.models.GPRegression(X, phase_i.reshape(-1, 1), kernel, noise_var=cfg.noise_var)
        model = GPy.models.GPClassification(X, phase_i.reshape(-1, 1), kernel)

        if cfg.optim_step:
            model.optimize()
            var, r = float(kernel.variance), float(kernel.lengthscale)
            if r > 10:
                kernel.lengthscale = 10
            if r < 0.1:
                kernel.lengthscale = 0.1
            if var < 1:
                kernel.var = 1

            var, r = float(kernel.variance), float(kernel.lengthscale)
            print(f'{var = :.2g}, {r = :.2g}')

        models.append(model)

    return models


def dist2(pd1s: np.ndarray, pd2s: np.ndarray, weights=None):
    # n_diagrams, n_points = pd1s.shape

    diffs = np.not_equal(pd1s, pd2s)
    mean_diffs = np.mean(diffs, axis=1)  # Mean over each phase diagram
    pd_diffs = mean_diffs

    if weights is not None:
        mean_diffs *= weights
        mean_diffs = np.sum(mean_diffs)
    else:
        mean_diffs = np.mean(mean_diffs)

    return mean_diffs


# Probability single gaussian is larger than the rest by integral of form f(x) exp(-x^2) where f(x) is product of CDFs.
def max_1(mu_1, sigma_1, mus, sigmas, hermite_roots):
    xs, weights = hermite_roots  # x to sample, weights for summation

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
    """Probability of observing each phase at x_{n+1}. Everything is gaussian, so this is finding the max of gaussians."""
    mus, vars = [], []
    for model in models:
        mu, var = model.predict(x_new.reshape(-1, 2), include_likelihood=False)
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
def gen_pd_new_point(models: list[GPy.core.GP], x_new, sample_xs, cfg):
    """Sample P_{n+1}(x_{n+1}, y), new phase diagrams assuming a new observation is taken at x_new.
    Returns: Phase diagrams, maximum probability of observing"""

    # First sample P_{n}(y_{n+1})
    pd_probs = sample_new_y(models, x_new)

    # Skip sampling a point if certainty is already very high
    if max(pd_probs) > cfg.skip_point:
        return None, pd_probs, None

    # Sample new models for each possible observation
    pds, sampled_phase = [], []
    for obs_phase, obs_prob in enumerate(pd_probs):
        # Ignore very unlikely observations that will have 0 samples
        if obs_prob < cfg.skip_phase:
            pds.append(np.empty((0, sample_xs.shape[0])))
            continue

        new_models = []
        for phase_i, model in enumerate(models):
            kern_var, kern_len = float(model.kern.variance), float(model.kern.lengthscale)
            X, Y = model.X, model.Y

            y_new = int(phase_i == obs_phase)
            X_new, Y_new = np.vstack([X, x_new]), np.vstack([Y, y_new])
            kernel = GPy.kern.Matern52(input_dim=2, variance=kern_var, lengthscale=kern_len)
            # model = GPy.models.GPRegression(X, phase_i.reshape(-1, 1), kernel, noise_var=cfg.noise_var)
            model = GPy.models.GPClassification(X_new, Y_new, kernel)


            new_models.append(model)


        # Sample new phase diagrams, weighted to probability model is observed.
        if cfg.sample_new is None:
            # Use probability weighting
            pd_new = gen_pd(new_models, sample_xs, sample=None, cfg=cfg)
        else:
            # Use number of phase diagrams as weighting
            pd_new = gen_pd(new_models, sample_xs, sample=round(cfg.sample_new * obs_prob), cfg=cfg)  # Note, rounding here.

        pds.append(pd_new)
        sampled_phase.append(obs_phase)

    pds = np.concatenate(pds)
    return pds, pd_probs, sampled_phase


# Compute A(x) over all points
def acquisition(models, new_Xs, cfg):
    # Grid over which to compute distances
    X_dist, X1_dist, X2_dist = make_grid(cfg.N_dist, [0, 1, 0, 1])
    # P_n
    pd_old = gen_pd(models, X_dist, sample=cfg.sample_old, cfg=cfg)

    # P_{n+1}
    avg_dists, full_probs = [], []
    for new_X in new_Xs:
        if cfg.sample_dist is not None:
            # Sample distance a region around selected point only
            X_min = new_X - cfg.sample_dist / 2
            X_max = X_min + cfg.sample_dist

            X_min = np.maximum(X_min, [0, 0])
            X_max = np.minimum(X_max, [1, 1])

            # Create a mask of wanted points
            mask = (X_dist[:, 0] >= X_min[0]) & (X_dist[:, 0] <= X_max[0]) & (X_dist[:, 1] >= X_min[1]) & (X_dist[:, 1] <= X_max[1])

            pd_old_want = pd_old[:, mask]
            X_dist_region = X_dist[mask]

        else:
            X_dist_region = X_dist
            pd_old_want = pd_old

        pd_new, pd_probs, sampled_phase = gen_pd_new_point(models, new_X, sample_xs=X_dist_region, cfg=cfg)
        full_probs.append(pd_probs)

        if pd_new is None:  # Skipped sampling point
            avg_dists.append(0)
            continue

        # Expected distance between PD1 and PD2 averaged over all pairs
        pd_old_repeat = np.repeat(pd_old_want, pd_new.shape[0], axis=0)
        pd_new_tile = np.tile(pd_new, (pd_old_want.shape[0], 1))

        if cfg.sample_new is None:
            weights = np.tile(pd_probs, pd_old_want.shape[0])
            avg_dist = dist2(pd_old_repeat, pd_new_tile, weights=weights[sampled_phase])
        else:
            avg_dist = dist2(pd_old_repeat, pd_new_tile)
        avg_dists.append(avg_dist)

    X_display, _, _ = make_grid(cfg.N_display, [0, 1, 0, 1])
    pd_old_mean = gen_pd(models, X_display, sample=None, cfg=cfg)[0]

    return pd_old_mean, np.array(avg_dists), np.stack(full_probs)


# Suggest a single point to sample given current observations
def suggest_point(obs_holder, cfg):
    # Fit to existing observations
    models = fit_gp(obs_holder, cfg=cfg)

    # Find max_x A(x)
    new_Xs, _, _ = make_grid(cfg.N_eval, [0, 1, 0, 1])  # Points to test for aquisition

    pd_old, avg_dists, pd_probs = acquisition(models, new_Xs, cfg)
    max_pos = np.argmax(avg_dists)
    new_point = new_Xs[max_pos]
    prob_at_point = pd_probs[max_pos]

    # Rescale new point to real coordiantes
    new_point = to_real_scale(new_point, cfg.extent)

    return new_point, (pd_old, avg_dists, pd_probs), prob_at_point


# Sample two points
def suggest_two_points(obs_holder, cfg):
    new_point, plot_probs, prob_at_point = suggest_point(obs_holder, cfg)
    obs_holder.make_obs(new_point)
    sec_point, _, _ = suggest_point(obs_holder, cfg)

    return new_point, sec_point, plot_probs
