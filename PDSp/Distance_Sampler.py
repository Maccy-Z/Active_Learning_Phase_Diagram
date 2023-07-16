# import edit_source_files
import GPy
import numpy as np
import time
import scipy
from matplotlib import pyplot as plt

from utils import ObsHolder, make_grid, plot_scale, new_save_folder
from config import Config

np.set_printoptions(precision=2)
cfg = Config()
N_phases = 3
save_dir = "./saves"

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
def fit_gp(obs_holder: ObsHolder) -> list[GPy.core.GP]:
    "Trains a model for each phase."
    X, Y = obs_holder.get_obs()
    # X, _, _ = make_grid(cfg.N_train)
    # true_pd = tri_pd(X)

    models = []
    for i in range(N_phases):
        phase_i = (Y == i) * 2 - 1  # Between -1 and 1

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
            kern_var, kern_len = model.kern.variance, model.kern.lengthscale
            print()
            print()
            print()

            #print(np.show_config())
            X, Y = model.X, model.Y

            y_new = 2 * int(phase_i == obs_phase) - 1
            X_new, Y_new = np.vstack([X, x_new, x_new]), np.vstack([Y, y_new, y_new])

            kernel = GPy.kern.Matern52(input_dim=2, variance=kern_var, lengthscale=kern_len)

            print()
            A = model.kern.K(X_new)
            print(A)
            print()

            model = GPy.models.GPRegression(X_new, Y_new, kernel, noise_var=0.01)
            #model.optimize()
            new_models.append(model)
            exit(5)

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


def acquisition(models, new_Xs):
    # Grid over which to compute distances
    X, _, _ = make_grid(cfg.N_dist, xmin=cfg.xmin, xmax=cfg.xmax)

    # P_n
    pd_old = gen_pd(models, X, sample=cfg.sample_old)

    # P_{n+1}
    avg_dists = []
    for new_X in new_Xs:
        print(new_X, end=" ")
        pd_new = gen_pd_new_point(models, new_X, sample_xs=X, sample=cfg.sample_new)

        # Expected distance between PD1 and PD2 averaged over all pairs
        pd_old_repeat = np.repeat(pd_old, pd_new.shape[0], axis=0)
        pd_new_tile = np.tile(pd_new, (pd_old.shape[0], 1))
        avg_dist = dist2(pd_old_repeat, pd_new_tile)
        avg_dists.append(avg_dist)


    return pd_old[0], avg_dists


def main():
    obs_holder = ObsHolder(Config())
    save_path = new_save_folder(save_dir)

    # Init observations to start off
    X_init, _, _ = make_grid(cfg.N_init)
    for xs in X_init:
        obs_holder.make_obs(xs)

    for i in range(30):
        st = time.time()

        # Fit to existing observations
        models = fit_gp(obs_holder)

        # Find max_x A(x)
        new_Xs, _, _ = make_grid(cfg.N_eval, xmin=cfg.xmin, xmax=cfg.xmax)  # Points to test for aquisition

        pd_old, avg_dists = acquisition(models, new_Xs)
        max_pos = np.argmax(avg_dists)
        new_point = new_Xs[max_pos]
        obs_holder.make_obs(new_point)

        print()
        print()
        print("New point to sample from:", new_point)

        # Plot current phase diagram and next sample point
        xs_train, ys_train = models[0].X[:, 0], models[0].X[:, 1]
        plt.title(f"PD and sample points {i}")
        plt.imshow(pd_old.reshape(cfg.N_dist, cfg.N_dist), extent=(-2, 2, -2, 2), origin="lower")   # Phase diagram
        plt.scatter(xs_train, ys_train, marker="x", s=40)   # Existing observations
        plt.scatter(new_point[0], new_point[1], s=100)      # New observations

        plt.savefig(f'{save_path}/{i}.png', bbox_inches="tight")
        plt.show()

        print(f'Iter: {i}, Time: {time.time() - st:.3g}s ' )
        print()
        # avg_dists = np.array(avg_dists).reshape(cfg.N_eval, cfg.N_eval)

        # plt.imshow(avg_dists)
        # plt.xlim([0, cfg.N_eval - 1]), plt.ylim([0, cfg.N_eval - 1])
        #
        # plt.title(new_point)
        # plt.colorbar()
        # plt.show()

    obs_holder.plot_samples()
    # with open("./saves/both_sample", "wb") as f:
    #     pickle.dump(dists, f)




if __name__ == "__main__":

    main()


