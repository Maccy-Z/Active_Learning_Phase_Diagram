# Evaluate error of GP model
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
        y_pred, _ = pd_model.predict(xs, include_likelihood=False)  # m.shape = [n**2, 1]

        y_preds.append(y_pred)

    y_preds = np.stack(y_preds)
    sample_pds = np.argmax(y_preds, axis=0).T
    return None, None, sample_pds


# Distance between true PD and prediction
def dist(obs_holder, *, true_pd, cfg, points, t):
    # Observations up to time t
    T = t + 10
    Xs, Ys = obs_holder.get_og_obs()

    obs_holder._obs_pos = Xs[:T]
    obs_holder.obs_phase = Ys[:T]

    # plot_Xs, X1, X2 = make_grid(points, cfg.extent)
    model_Xs, _, _ = make_grid(points, (0, 1, 0, 1))
    models = fit_gp(obs_holder, cfg=cfg)
    pds = single_pds(models, model_Xs)[2].reshape(points, points)
    pds = np.flip(np.rot90(np.rot90(pds)), axis=1)
    true_pd = np.stack(true_pd).reshape(points, points)

    diff = np.not_equal(pds, true_pd)
    diff_mean = np.mean(diff)

    # print("\033[91mDelete this to stop plotting\033[0m")
    # plt.imshow(pds)
    # plt.show()
    #exit(5)
    return diff_mean


def deduplicate(array_list):
    seen = []  # List to store the distinct arrays seen so far
    count_dict = {}  # Dictionary to store the result

    count = 0
    for idx, arr in enumerate(array_list):
        # Check if the current array is identical to any of the arrays seen so far
        if not any(np.array_equal(arr, seen_arr) for seen_arr in seen):
            seen.append(arr)
            count += 1
        count_dict[idx] = count

    reversed_dict = {}
    for key, value in count_dict.items():
        if value not in reversed_dict:
            reversed_dict[value] = key
    return reversed_dict

def main():
    true_pd = np.array([[2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2.],
                        [1., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2.],
                        [1., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2.],
                        [1., 1., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2.],
                        [1., 1., 1., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2.],
                        [1., 1., 1., 1., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2.],
                        [1., 1., 1., 1., 1., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2.],
                        [1., 1., 1., 1., 1., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2.],
                        [1., 1., 1., 1., 1., 1., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2.],
                        [1., 1., 1., 1., 1., 1., 1., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2.],
                        [1., 1., 1., 1., 1., 1., 1., 1., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2.],
                        [1., 1., 1., 1., 1., 1., 1., 1., 1., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2.],
                        [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2.],
                        [0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 2., 2., 2., 2., 2., 2., 2., 2., 2.],
                        [0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 2., 2., 2., 2., 2., 2., 2., 2.],
                        [0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 2., 2., 2., 2., 2.],
                        [0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 2.],
                        [0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                        [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1.],
                        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])
    eval_points = 21


    assert true_pd != [], ("Fill in the true phase diagram as a 2D numpy array, with the same number of points as eval_points. "
                           "It might need transposing / reflecting to orient properly. ")
    plt.imshow(true_pd)
    plt.show()

    f = sorted([int(s) for s in os.listdir("./saves")])

    save_name = f[-1]
    print(f'{save_name = }')

    og_obs = ObsHolder.load(f'./saves/{save_name}')

    # Deduplicate observations
    all_obs = og_obs._obs_pos
    obs_map = deduplicate(all_obs)

    with open(f'./saves/{save_name}/cfg.pkl', "rb") as f:
        cfg = pickle.load(f)

    errors = []
    for t in range(len(og_obs.obs_phase)):
        obs_holder = copy.deepcopy(og_obs)
        error = dist(obs_holder, true_pd=true_pd, points=eval_points, t=t, cfg=cfg)
        errors.append(error)
    plt.plot(errors)
    plt.show()
    #
    # print()
    # for s in [10, 20, 30, 40, 50]:
    #     print(f'{errors[s]}')

    print("Errors:")
    print("Copy me to error_plot.py")
    print(errors)
    print("Map number of observations made to number of unique observations (remove duplicates)")
    print("Copy me to error_plot.py")
    print(obs_map)


if __name__ == "__main__":
    main()
