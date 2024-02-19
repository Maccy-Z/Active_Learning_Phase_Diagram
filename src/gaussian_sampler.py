import numpy as np
from utils import ObsHolder, make_grid, to_real_scale, points_within_radius, c_print
from config import Config
from GP import GPRSoftmax
from matplotlib import pyplot as plt
import time

np.set_printoptions(precision=2)


class DistanceCalculator:
    def __init__(self, probs_old, cfg: Config):
        # probs_old.shape = [N_samples ** 2, N_phases]
        self.cfg = cfg

        pds_old = np.argmax(probs_old, axis=1)  # Shape = [N_samples ** 2]

        # Tile since acqusition function is computed over all possible observations
        pds_old = np.tile(pds_old, (cfg.N_phases, 1))  # Shape = [N_phases, N_samples ** 2]
        probs_old = np.tile(probs_old, (cfg.N_phases, 1, 1))  # Shape = [N_phases, N_samples ** 2, N_phases]

        self.pds_old = pds_old
        self.probs_old = probs_old

        self.dist = self.abs_change

    def abs_change(self, probs_new, dist_mask, sampled_mask, weights):
        """
        probs_new.shape = [n_phases, N_dist, N_phases]. First dim is search phases, last is probs for each phase
        """
        pds_new = np.argmax(probs_new, axis=-1)  # Shape = [n_phases, N_dist]
        pds_old = self.pds_old[sampled_mask][:, dist_mask]  # Shape = [n_phases, N_dist]

        diffs = np.not_equal(pds_old, pds_new)
        mean_diffs = np.sum(diffs, axis=1).astype(float)  # Mean over each phase diagram
        mean_diffs *= weights[sampled_mask]
        mean_diffs = np.sum(mean_diffs)

        return mean_diffs


def fit_gp(obs_holder: ObsHolder, cfg) -> GPRSoftmax:
    "Fit a new model to observations, including hyperparam tuning "
    model = GPRSoftmax(obs_holder, cfg)
    model.make_GP(fit_hyperparams=True)

    return model


# Sample phase diagrams from models.
def gen_pd(model: GPRSoftmax, xs, cfg: Config):
    """
    @param model: Models
    @param xs: Points to sample from. Shape = [N_samples ** 2, N_phases]
    @param cfg: Config
    @return: Most likely phase diagram.
    """
    probs = model.predict(xs)  # shape = [N_samples ** 2, N_phases]
    # pred_ys = np.argmax(probs, axis=1)

    return probs  # , pred_ys


# Predict new observaion and phase diagram
def gen_pd_new_point(old_model: GPRSoftmax, x_new, sample_xs, obs_probs, cfg: Config):
    """Sample P_{n+1}(x_{n+1}, y), new phase diagrams assuming a new observation is taken at x_new.
    Returns: Phase diagrams and which phases were sampled"""

    # Sample new models for each possible observation
    sampled_mask, probs = [], []
    for obs_phase, obs_prob in enumerate(obs_probs):
        # Ignore very unlikely observations
        if obs_prob < cfg.skip_phase:
            sampled_mask.append(False)
            continue
        sampled_mask.append(True)

        # Make fantasy model assuming new observation is taken
        fant_model = old_model.fantasy_GPs(x_new, obs_phase, cfg.obs_prob)

        # Sample new phase diagrams, weighted to probability model is observed.
        pred_probs = gen_pd(fant_model, sample_xs, cfg=cfg)

        probs.append(pred_probs)
        # prob_mean = np.array(prob_mean)
        # print(prob_mean.shape)

        # plt.close("all")
        # plt.subplot(1, 3, 1)
        # plt.title(f"Observed Phase {obs_phase}")
        # plt.imshow(prob_mean[:, 0].reshape([cfg.N_dist, cfg.N_dist]).T, origin="lower", extent=(0, 1, 0, 1), vmax=1, vmin=0.0)
        # plt.scatter(x_new[0], x_new[1], c="orange")
        # plt.scatter(X_old[:, 0], X_old[:, 1], c=y_old)
        #
        # plt.subplot(1, 3, 2)
        # plt.imshow(prob_std[:, 0].reshape([cfg.N_dist, cfg.N_dist]).T, origin="lower", extent=(0, 1, 0, 1), vmax=0.3, vmin=0)
        # plt.scatter(X_old[:, 0], X_old[:, 1], c=y_old)
        # plt.scatter(x_new[0], x_new[1], c="orange")
        #
        # plt.subplot(1, 3, 3)
        # plt.imshow(pred_ys.reshape([cfg.N_dist, cfg.N_dist]).T, origin="lower", extent=(0, 1, 0, 1), vmax=1., vmin=0)
        # plt.scatter(X_old[:, 0], X_old[:, 1], c=y_old)
        # plt.scatter(x_new[0], x_new[1], c="orange")
        # plt.show()

    probs = np.stack(probs)  # shape = [n_phase, N_samples ** 2, N_phases]
    return probs, sampled_mask


def search_point(old_model: GPRSoftmax, new_X, X_dist, obs_prob, dist_calc, cfg):
    # Skip point if certain enough
    if np.max(obs_prob) > cfg.skip_point:
        return 0.

    # Sample distance a region around selected point only
    mask = points_within_radius(X_dist, new_X, cfg.sample_dist, cfg.unit_extent)
    X_dist_region = X_dist[mask]

    pd_new, sampled_phase = gen_pd_new_point(old_model, new_X, sample_xs=X_dist_region, obs_probs=obs_prob, cfg=cfg)

    # Expected distance between PD1 and PD2 averaged over all pairs
    avg_dist = dist_calc.dist(pd_new, dist_mask=mask, sampled_mask=sampled_phase, weights=obs_prob)

    return avg_dist


# Compute A(x) over all points
def acquisition(old_model: GPRSoftmax, acq_Xs, pool, cfg):
    # Grid over which to compute distances
    X_dist, _ = make_grid(cfg.N_dist, cfg.unit_extent)

    # P_n
    probs_old = gen_pd(old_model, X_dist, cfg=cfg)  # For computing distances
    obs_probs = gen_pd(old_model, acq_Xs, cfg=cfg)  # Probs for sampling A(x)
    dist_calc = DistanceCalculator(probs_old, cfg)

    # P_{n+1}
    args = [(old_model, new_X, X_dist, obs_prob, dist_calc, cfg) for new_X, obs_prob in zip(acq_Xs, obs_probs)]
    avg_dists = pool.starmap(search_point, args)

    return np.array(avg_dists), obs_probs


# Suggest a single point to sample given current observations
def suggest_point(pool, obs_holder, cfg: Config):
    # Fit to existing observations
    model = fit_gp(obs_holder, cfg=cfg)

    # Find max_x A(x)
    acq_Xs, _ = make_grid(cfg.N_eval, cfg.unit_extent)  # Points to test for aquisition
    # acq_Xs = np.array([[0.5, 0.5], [0.5, 0.5]])

    acq_fn, pd_probs = acquisition(model, acq_Xs, pool, cfg)
    max_pos = np.argmax(acq_fn)

    new_point = acq_Xs[max_pos]
    prob_at_point = pd_probs[max_pos]

    # Rescale new point to real coordiantes
    # new_point = to_real_scale(new_point, cfg.extent)

    # Plot old phase diagram
    X_display, _ = make_grid(cfg.N_display, cfg.unit_extent)
    probs_old = gen_pd(model, X_display, cfg=cfg)
    pd_old = np.argmax(probs_old, axis=1)

    return new_point, prob_at_point, (pd_old, acq_fn, pd_probs),


