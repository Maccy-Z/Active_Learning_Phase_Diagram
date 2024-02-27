# Evaluate error of GP model
import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

cwd = os.getcwd()
os.chdir(os.path.dirname(cwd))
print(os.getcwd())

import pickle
import numpy as np
from matplotlib import pyplot as plt
import copy

from src.utils import make_grid, ObsHolder, tri_pd, bin_pd, quad_pd
from src.config import Config
from src.gaussian_sampler import fit_gp, gen_pd


# Distance between true PD and prediction
def dist(obs_holder: ObsHolder, *, pd_fn, cfg, points, t):
    # Truncate observations up to time t, assuming first 2 observations are given.
    Xs, Ys, prob = obs_holder.get_og_obs()
    T = t + 2
    obs_holder._obs_pos = Xs[:T]
    obs_holder.obs_phase = Ys[:T]
    obs_holder.obs_prob = prob[:T]

    model = fit_gp(obs_holder, cfg=cfg)

    X_eval, _ = make_grid(points, cfg.unit_extent)
    probs = gen_pd(model, X_eval, cfg=cfg)
    pd_pred = np.argmax(probs, axis=1).reshape((points,) * cfg.N_dim)

    x_eval_real, _ = make_grid(points, cfg.extent)
    true_pd = []
    for X in x_eval_real:
        true_phase, _ = pd_fn(X, train=False)
        true_pd.append(true_phase)
    pd_true = np.stack(true_pd).reshape((points,) * cfg.N_dim)

    diff = np.not_equal(pd_pred, pd_true)
    diff_mean = np.mean(diff)

    return diff_mean


def main():
    pd_fn = quad_pd
    f = sorted([int(s) for s in os.listdir("./saves")])

    save_name = f[-1]
    print(f'{save_name = }')

    og_obs = ObsHolder.load(f'./saves/{save_name}')

    with open(f'./saves/{save_name}/cfg.pkl', "rb") as f:
        cfg: Config = pickle.load(f)

    errors = []
    for t in range(len(og_obs.obs_phase)):
        print()
        obs_holder = copy.deepcopy(og_obs)
        error = dist(obs_holder, pd_fn=pd_fn, points=19, t=t, cfg=cfg)
        errors.append(error)
    plt.plot(errors)
    plt.ylim([0.0, 0.2])
    plt.show()
    #
    # print()
    # for s in [10, 20, 30, 40, 50]:
    #     print(f'{errors[s]}')

    print()
    print(errors)


if __name__ == "__main__":
    main()
