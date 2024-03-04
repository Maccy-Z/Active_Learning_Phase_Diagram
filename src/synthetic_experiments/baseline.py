# Evaluate error of GP model
import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

import GPy
import numpy as np
from matplotlib import pyplot as plt

from my_code.utils import ObsHolder, make_grid, bin_pd
from my_code.config import Config


# Fit model to existing observations
def fit_gp(obs_holder: ObsHolder, cfg=None, t=None) -> list[GPy.core.GP]:
    "Trains a model for each phase."
    X, Y = obs_holder.get_obs()

    Y = Y * 2 - 1
    kernel = GPy.kern.Matern52(input_dim=2)
    model = GPy.models.GPRegression(X, Y.reshape(-1, 1), kernel, noise_var=2)

    model.optimize()
    var, r = float(kernel.variance), float(kernel.lengthscale)

    if r < 0.1:
        kernel.lengthscale = 0.1
    # print(f'{var = :.2g}, {r = :.2g}')

    return model


# Sample phase diagrams from models.
def gen_pd(models: list[GPy.core.GP], xs):
    """
    @param models: List of models for each phase
    @param xs: Points to sample from
    @param sample: Use mean or sample from GP.
    @return: List of possible phase diagrams.
    """
    y_mean, y_var = models.predict(xs)  # m.shape = [n**2, 1]

    return y_mean, y_var, y_mean > 0


# Compute A(x) over all points
def acquisition(models, new_Xs, cfg):
    # Grid over which to compute distances

    y_mean, y_var, _ = gen_pd(models, new_Xs)
    y_mean, y_var = y_mean.reshape(19, 19), y_var.reshape(19, 19)

    a = - np.abs(y_mean) / (np.sqrt(y_var) + 5e-2)

    return a, y_mean, y_var


def suggest_point(obs_holder, cfg):
    models = fit_gp(obs_holder)

    # Find max_x A(x)
    new_Xs, _, _ = make_grid(cfg.N_eval, cfg.extent)  # Points to test for aquisition

    acq, mean, var = acquisition(models, new_Xs, cfg)
    max_pos = np.argmax(acq)
    new_point = new_Xs[max_pos]

    return new_point, (acq, mean, var)


def plot(new_point, plots, obs_holder, cfg):
    # print(f'{new_point = }')

    fig, axs = plt.subplots(1, 3, figsize=(9, 3))
    axs[0].set_title("Acquisition")
    axs[1].set_title("Mean")
    axs[2].set_title("Var")

    for plot, ax in zip(plots, axs):
        cax = ax.imshow(plot, extent=cfg.extent, origin="lower")

        X_obs, phase_obs = obs_holder.get_obs()
        xs_train, ys_train = X_obs[:, 0], X_obs[:, 1]
        ax.scatter(xs_train, ys_train, marker="x", s=10, c=phase_obs, cmap='bwr')
        ax.scatter(new_point[0], new_point[1], s=30)
        fig.colorbar(cax, ax=ax, shrink=0.9)

        ax.spines['top'].set_visible(False)

    plt.subplots_adjust(wspace=0.4)
    plt.tight_layout()
    plt.show()


def error(pred_pd):
    plot_Xs, X1, X2 = make_grid(pred_pd.shape[0], Config().extent)
    true_pd = []
    for X in plot_Xs:
        true_phase = bin_pd(X, train=False)
        true_pd.append(true_phase)

    true_pd = np.stack(true_pd).reshape(pred_pd.shape)

    diff = np.not_equal(pred_pd, true_pd)
    diff_mean = np.mean(diff)
    return diff_mean


def main():
    cfg = Config()
    # save_path = new_save_folder("./saves")
    obs_holder = ObsHolder(cfg, "./tmp")

    # X_init, _, _ = make_grid(4, cfg.extent)
    X_init = [[0, -1.2], [0, 1]]
    phase_init = np.array([bin_pd(X) for X in X_init])

    for X, Y in zip(X_init, phase_init):
        obs_holder.make_obs(X, phase=Y)

    errors = []
    for i in range(51):
        # print("Step", i)
        new_point, plots = suggest_point(obs_holder, cfg)
        obs_holder.make_obs(new_point, phase=bin_pd(new_point))

        y_pred = plots[1] > 0
        e = error(y_pred)
        errors.append(e)
        if i % 10 == 0:
            #plot(new_point, plots, obs_holder, cfg)

            print(e)

    print(errors)

    #obs_holder.save()


if __name__ == "__main__":
    main()
