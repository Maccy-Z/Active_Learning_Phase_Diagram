# Plot final model predictions
import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

cwd = os.getcwd()
os.chdir(os.path.dirname(cwd))

import pickle
import GPy
import numpy as np
import math
from matplotlib import pyplot as plt
import copy

from utils import make_grid, ObsHolder, tri_pd, bin_pd, quad_pd
from config import Config
from gaussian_sampler import fit_gp


# Sample phase diagrams from models.
def single_pds(models: list[GPy.core.GP], xs):
    """
    @param models: List of models for each phase
    @param xs: Points to sample from
    @param sample: Use mean or sample from GP.
    @return: List of possible phase diagrams.
    """
    y_preds = []
    for phase_i, pd_model in enumerate(models):
        y_pred, _ = pd_model.predict(xs, include_likelihood=True)  # m.shape = [n**2, 1]

        y_preds.append(y_pred)

    y_preds = np.stack(y_preds)
    sample_pds = np.argmax(y_preds, axis=0).T
    return None, None, sample_pds


# Distance between true PD and prediction
def dist(obs_holder, *, pd_fn, cfg, points, t, ax):
    # Observations up to time t
    T = t + 2
    Xs, Ys = obs_holder.get_og_obs()

    obs_holder._obs_pos = Xs[:T]
    obs_holder.obs_phase = Ys[:T]

    plot_Xs, X1, X2 = make_grid(points, cfg.extent)
    model_Xs, _, _ = make_grid(points, (0, 1, 0, 1))
    models = fit_gp(obs_holder, cfg=cfg)
    pds = single_pds(models, model_Xs)[2].reshape(points, points)

    print(cfg.extent)
    xmin, xmax, ymin, ymax = cfg.extent
    ax.set_title(f'Ours', fontsize=12)
    ax.imshow(pds, origin="lower", extent=cfg.extent, aspect='auto')
    ax.scatter(Xs[:T, 0], Xs[:T, 1], marker="x", s=30, c=Ys[:T], cmap='bwr')  # Existing observations

    ax.set_xticks(np.linspace(xmin, xmax, 3), labels=np.linspace(xmin, xmax, 3), fontsize=12)
    ax.set_yticks(np.linspace(ymin, ymax, 3), labels=np.linspace(ymin, ymax, 3), fontsize=11)


def main():
    pd_fn = quad_pd
    f = sorted([int(s) for s in os.listdir("./saves")])

    save_name = f[-1]
    print(f'{save_name = }')

    og_obs = ObsHolder.load(f'./saves/{save_name}')

    with open(f'./saves/{save_name}/cfg.pkl', "rb") as f:
        cfg = pickle.load(f)

    fig, ax = plt.subplots(figsize=(3, 3))
    plt.subplots_adjust(left=0.175, right=0.95, top=0.925, bottom=0.1, wspace=0.4, hspace=0.4)

    errors = []
    t = len(og_obs.obs_phase)
    print("Total number of observations:", t)
    obs_holder = copy.deepcopy(og_obs)
    dist(obs_holder, pd_fn=pd_fn, points=19, t=t, cfg=cfg, ax=ax)

    #fig.tight_layout()

    fig.show()


if __name__ == "__main__":
    main()
