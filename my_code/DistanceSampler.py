# Code to plot the results. Run directly for automatic evaluation
# Workaround for multiprocesssing with torch ops
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
import time
import torch

from gaussian_sampler import suggest_point
from utils import ObsHolder, new_save_folder, tri_pd
from config import Config


class DistanceSampler:
    def __init__(self, init_phases, init_Xs, cfg: Config, save_dir="./saves"):
        save_path = new_save_folder(save_dir)
        print(f'{save_path = }')
        print()

        self.cfg = cfg
        self.obs_holder = ObsHolder(self.cfg, save_path=save_path)

        # Check if initial inputs are within allowed area
        points = np.array(init_Xs)
        if points.shape[1] != self.cfg.N_dim:
            raise ValueError(f"Number of dimensions must be {self.cfg.N_dim}, got {points.shape[1]}")
        bounds = np.array(self.cfg.extent)
        mins, maxs = bounds[:, 0], bounds[:, 1]

        all_in = np.logical_and(points >= mins, points <= maxs).all(axis=1)
        if not all_in.all():
            print("\033[31mWarning: Observations are outside search area\033[0m")

        # Check phases are allowed
        for phase, X in zip(init_phases, init_Xs, strict=True):
            self.obs_holder.make_obs(X, phase=phase)

        self.pool = torch.multiprocessing.Pool(processes=self.cfg.N_CPUs)

    def plot(self, plot_holder):
        raise NotImplementedError

    def add_obs(self, phase: int, X: np.ndarray):
        self.obs_holder.make_obs(X, phase=phase)
        self.obs_holder.save()

    def single_obs(self):
        new_point, prob_at_point, _ = suggest_point(self.pool, self.obs_holder, self.cfg)
        # self.plot(new_point, pd_old, avg_dists, pd_probs)

        return new_point, prob_at_point

    def __del__(self):
        # Safely close the pool when the class instance is deleted
        self.pool.close()
        self.pool.join()


def main(save_dir):
    pd_fn = tri_pd

    print(save_dir)
    cfg = Config()

    # Init observations to start off
    X_init = [[0.0, 1.], [0, -1]]
    phase_init = [pd_fn(X) for X in X_init]

    distance_sampler = DistanceSampler(phase_init, X_init, cfg, save_dir=save_dir)

    for i in range(51):
        print()
        print("Step:", i)
        st = time.time()
        new_point, probs = distance_sampler.single_obs()
        new_point, probs = new_point.tolist(), probs.tolist()

        print(f'Time taken: {time.time() - st:.1f}s')
        print(f'new point: {[round(f, 2) for f in new_point]}, probs: {[round(f, 2) for f in probs]}')

        obs_phase = pd_fn(new_point)
        distance_sampler.add_obs(obs_phase, new_point)



if __name__ == "__main__":
    save_dir = "./saves"

    main(save_dir)
