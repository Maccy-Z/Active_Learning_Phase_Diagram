import numpy as np
from utils import ObsHolder, make_grid, to_real_scale, points_within_radius, c_print
from config import Config
from GP import GPRSoftmax
from matplotlib import pyplot as plt

np.set_printoptions(precision=2)


class DistanceCalculator:
    def __init__(self, preds_old, cfg: Config):
        self.cfg = cfg

        probs, pds = preds_old
        pds = np.tile(pds, (cfg.N_phases, 1))  # Shape = [N_phases, N_samples ** 2]
        probs = np.tile(probs, (cfg.N_phases, 1, 1))  # Shape = [N_phases, N_samples ** 2, N_phases]

        self.pds = pds
        self.prob_means = probs

        self.dist = self.abs_change

    def abs_change(self, pds_new, dist_mask, sampled_mask, weights):
        # pd.shape = [n_phases, N_samples ** 2]
        probs_new, pds_new = pds_new

        pds_old = self.pds[sampled_mask][:, dist_mask]
        prob_means_old = self.prob_means[sampled_mask][:, dist_mask]

        # print(pds_old.shape, pds_new.shape)
        diffs = np.not_equal(pds_old, pds_new)
        mean_diffs = np.sum(diffs, axis=1).astype(float)  # Mean over each phase diagram
        mean_diffs *= weights
        mean_diffs = np.sum(mean_diffs)

        return mean_diffs

    def KL_div(self, pds_new, dist_mask, sampled_mask, weights):
        """
        KL div between PD_old and PD_new, assuming probs are independent Gaussian distributions.
        Sum over phase diagram, average over possible observations
        """

        pds_new, prob_means_new, prob_stds_new = pds_new
        pds_old = self.pds[sampled_mask][:, dist_mask]
        prob_means_old = self.prob_means[sampled_mask][:, dist_mask]
        prob_stds_old = self.prob_stds[sampled_mask][:, dist_mask]

        # prob_stds_old += 0.05
        # prob_stds_new += 0.05

        log_term = np.log(prob_stds_new / prob_stds_old)
        diff_term = (prob_stds_old ** 2 + (prob_means_old - prob_means_new) ** 2) / (2 * prob_stds_new ** 2)

        KL_div = log_term + diff_term - 0.5  # shape = (n_phase, num_dists, N_phases)
        # KL_div = (prob_means_old - prob_means_new) ** 2

        # Sum over phase diagrams
        KL_div = np.mean(KL_div, axis=(1, 2))  # shape = (n_phase,)

        KL_div *= weights
        distance = np.sum(KL_div)

        self.plot_means(prob_means_old, prob_means_new, distance)
        return distance

    def plot_means(self, old_means, new_means, distance):
        old = old_means[0, :, 0].reshape(11, 11)
        new_obs_0 = new_means[0, :, 0].reshape(11, 11)
        new_obs_1 = new_means[1, :, 0].reshape(11, 11)

        delta_0 = old - new_obs_0
        delta_1 = old - new_obs_1

        plt.close("all")
        plt.subplot(1, 3, 1)
        plt.title(distance)
        plt.imshow(old.T, origin="lower", extent=(0, 1, 0, 1), vmax=0.8, vmin=0.2)
        plt.scatter(0.5, 0.5, c='r')
        plt.subplot(1, 3, 2)
        plt.imshow(delta_0.T, origin="lower", extent=(0, 1, 0, 1), vmax=0.1, vmin=-0.1)
        plt.scatter(0.5, 0.5, c='r')
        plt.subplot(1, 3, 3)
        plt.imshow(delta_1.T, origin="lower", extent=(0, 1, 0, 1), vmax=0.1, vmin=-0.1)
        plt.scatter(0.5, 0.5, c='r')

        plt.show()


def fit_gp(fit_hyperparams: bool, obs_holder: ObsHolder, cfg) -> GPRSoftmax:
    "Fit a new model to observations"
    model = GPRSoftmax(obs_holder, cfg)
    hyperparams = model.make_GP(fit_hyperparams)

    # TODO: properly handle
    if fit_hyperparams:
        c_print(f"Hyperparams: {hyperparams}", "green")

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
    pred_ys = np.argmax(probs, axis=1)

    return probs, pred_ys


#
# def dist(pd_olds: np.ndarray, pds_new, weights):
#     # pd.shape = [n_phases, N_samples ** 2]
#     pds, prob_means, prob_stds = pds_new
#
#     diffs = np.not_equal(pd_olds, pds)
#     mean_diffs = np.mean(diffs, axis=1)  # Mean over each phase diagram
#
#     mean_diffs *= weights
#     mean_diffs = np.sum(mean_diffs)
#
#     return mean_diffs


# Predict new observaion and phase diagram
def gen_pd_new_point(old_model: GPRSoftmax, x_new, sample_xs, obs_probs, cfg):
    """Sample P_{n+1}(x_{n+1}, y), new phase diagrams assuming a new observation is taken at x_new.
    Returns: Phase diagrams and which phases were sampled"""

    # Sample new models for each possible observation
    pds, sampled_phase, probs = [], [], []
    for obs_phase, obs_prob in enumerate(obs_probs):
        # Ignore very unlikely observations
        if obs_prob < cfg.skip_phase:
            pds.append(np.empty((0, sample_xs.shape[0])))
            sampled_phase.append(0)
            continue

        # Make fantasy model assuming new observation is taken
        fant_model = old_model.fantasy_GPs(x_new, obs_phase)

        # Sample new phase diagrams, weighted to probability model is observed.
        pred_phase, pred_probs = gen_pd(fant_model, sample_xs, cfg=cfg)

        pds.append(pred_phase)
        probs.append(pred_probs)
        sampled_phase.append(1)

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

    pds = np.stack(pds)  # Shape = [n_phase, N_samples ** 2]
    probs = np.stack(probs)  # shape = [n_phase, N_samples ** 2, N_phases]

    return (pds, probs), sampled_phase


# Compute A(x) over all points
def acquisition(old_model: GPRSoftmax, acq_Xs, cfg):
    # Grid over which to compute distances
    X_dist, _ = make_grid(cfg.N_dist, cfg.unit_extent)

    # P_n
    probs_old, pd_old = gen_pd(old_model, X_dist, cfg=cfg)  # For computing distances
    obs_probs, _ = gen_pd(old_model, acq_Xs, cfg=cfg)  # Probs for sampling A(x)

    dist_calc = DistanceCalculator((probs_old, pd_old), cfg)

    # P_{n+1}
    avg_dists = []
    for new_X, obs_prob in zip(acq_Xs, obs_probs):
        # Skip point if certain enough
        if np.max(obs_prob) > cfg.skip_point:
            avg_dists.append(0)
            continue

        # Sample distance a region around selected point only
        mask = points_within_radius(X_dist, new_X, cfg.sample_dist, cfg.unit_extent)
        X_dist_region = X_dist[mask]

        pd_new, sampled_phase = gen_pd_new_point(old_model, new_X, sample_xs=X_dist_region, obs_probs=obs_prob, cfg=cfg)

        # Expected distance between PD1 and PD2 averaged over all pairs
        avg_dist = dist_calc.dist(pd_new, dist_mask=mask, sampled_mask=sampled_phase, weights=obs_prob)

        avg_dists.append(avg_dist)

    return np.array(avg_dists), obs_probs


# Suggest a single point to sample given current observations
def suggest_point(obs_holder, cfg: Config):
    # Fit to existing observations
    model = fit_gp(True, obs_holder, cfg=cfg)

    # Find max_x A(x)
    acq_Xs, _ = make_grid(cfg.N_eval, cfg.unit_extent)  # Points to test for aquisition
    # acq_Xs = np.array([[0.5, 0.5], [0.5, 0.5]])

    acq_fn, pd_probs = acquisition(model, acq_Xs, cfg)
    max_pos = np.argmax(acq_fn)
    new_point = acq_Xs[max_pos]
    prob_at_point = pd_probs[max_pos]

    # Rescale new point to real coordiantes
    new_point = to_real_scale(new_point, cfg.extent)

    # Plot old phase diagram
    X_display, _ = make_grid(cfg.N_display, cfg.unit_extent)
    pd_old_display = gen_pd(model, X_display, cfg=cfg)[1]

    return new_point, prob_at_point, (pd_old_display, acq_fn, pd_probs),


# Sample two points
def suggest_two_points(obs_holder, cfg):
    new_point, plot_probs, prob_at_point = suggest_point(obs_holder, cfg)
    obs_holder.make_obs(new_point)
    sec_point, _, _ = suggest_point(obs_holder, cfg)

    return new_point, sec_point, plot_probs
