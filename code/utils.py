# Helpful functions
from matplotlib import pyplot as plt
import numpy as np
import scipy
import os
import pickle
import shutil
import importlib.util

from config import Config


class ObsHolder:
    def __init__(self, cfg: Config, save_path):
        self.cfg = cfg
        self.save_path = save_path
        self._obs_pos = []
        self.obs_phase = []

    def make_obs(self, X, phase: int = None):
        self._obs_pos.append(X)
        # Test or use real data
        if phase is None:
            self.obs_phase.append(tri_pd(X))
        else:
            self.obs_phase.append(phase)

    def get_obs(self) -> [np.ndarray, np.ndarray]:
        # Rescale observations to [0, 1] on inference
        obs_pos = np.array(self._obs_pos)
        xmin, xmax, ymin, ymax = self.cfg.extent
        mins = np.array([xmin, ymin])
        maxs = np.array([xmax, ymax])

        X_range = maxs - mins

        obs_pos = (obs_pos - mins) / X_range
        return obs_pos, np.array(self.obs_phase)

    def get_og_obs(self):
        return np.array(self._obs_pos), np.array(self.obs_phase)

    def get_kern_param(self, t=None):
        if t is None:
            step = len(self.obs_phase)
        else:
            step = t

        kern_v, kern_r = self.cfg.kern_var, self.cfg.kern_r

        var = kern_v  # + ((step - 16) * .75 / 50)
        r = kern_r  # / (np.sqrt(step + 16)) + 0.25

        return var, r

    @staticmethod
    def load(save_path):
        # Load exactly from a save location, including functions
        # First load class functions. Then add in save history.
        spec = importlib.util.spec_from_file_location("module.name", f'{save_path}/utils.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        Cls = getattr(module, "ObsHolder")

        with open(f'{save_path}/obs_holder.pkl', "rb") as f:
            obs_holder = pickle.load(f)

        cls = Cls(obs_holder.cfg, obs_holder.save_path)
        cls._obs_pos = obs_holder._obs_pos
        cls.obs_phase = obs_holder.obs_phase

        return cls

    def save(self):
        with open(f'{self.save_path}/obs_holder.pkl', "wb") as f:
            pickle.dump(self, f)


# Test function, representing making an observation
def tri_pd(X):
    x, y = X[0], X[1]

    theta = np.arctan2(x, y) + 0.4
    theta = np.mod(theta, 2 * np.pi) - np.pi

    r = np.sqrt(x ** 2 + y ** 2)
    a = -np.pi
    b = -np.pi / 3
    c = np.pi / 3
    d = np.pi

    if theta < b or r < 1:
        return 0
    elif theta < c:
        return 1
    else:
        return 2


def bin_pd(X, train=True):
    x, y = X[0], X[1]

    if train:
        x += (np.random.rand() - 0.5) * 0.4
        y += (np.random.rand() - 0.5) * 0.4

    if y > 1 * np.sin(0.5 * np.pi * x):
        return 1
    else:
        return 0


def quad_pd(X, train=True):
    x, y = X[0], X[1]

    # Critical
    if x > 0.75 and y > 0.:
        return 3

    # Solid
    if x <= -0.6 and y >= 0.5 / 1.4 * x - 9 / 7:
        return 0

    # Liquid
    if -0.6 <= x <= 0.75 and y >= (10 / 27 * x ** 2 + 19 / 18 * x - 1):
        return 1

    # Gas
    return 2


# Nearest-neighbour plot
def plot_mean(obs_holder, show_obs=True):
    Xs = np.array(obs_holder.obs_pos)
    obs = np.array(obs_holder.obs_phase)

    if show_obs:
        plt.scatter(Xs[:, 0], Xs[:, 1], marker="x", s=40, c=obs, cmap='bwr')  # Existing observations

    # Interpolate points onto a grid
    xi = np.linspace(-2, 2, 100)  # x-coordinate of regular grid
    yi = np.linspace(-2, 2, 100)  # y-coordinate of regular grid
    xi, yi = np.meshgrid(xi, yi)  # Create grid mesh

    zi = scipy.interpolate.griddata(Xs, obs, (xi, yi), method='nearest')  # Interpolate using linear method

    plt.imshow(zi, origin="lower", extent=obs_holder.cfg.extent)
    # print(Xs[:, 0], Xs[:,1])
    # plt.tricontourf(Xs[:, 0], Xs[:,1], obs, levels=100)

    plt.xlim(obs_holder.cfg.xmin, obs_holder.cfg.xmax)
    plt.ylim(obs_holder.cfg.ymin, obs_holder.cfg.ymax)
    plt.show()


# Make nxn grid
def make_grid(n, extent) -> (np.ndarray, np.ndarray, np.ndarray):
    "Make a n x n grid or points"
    xmin, xmax, ymin, ymax = extent
    X1 = np.linspace(xmin, xmax, n)
    X2 = np.linspace(ymin, ymax, n)
    x1, x2 = np.meshgrid(X1, X2)
    X = np.hstack((x1.reshape(n ** 2, 1), x2.reshape(n ** 2, 1)))
    return X, X1, X2


# Rescale from physical coords to plot coords
def to_real_scale(X, extent):
    xmin, xmax, ymin, ymax = extent
    mins = np.array([xmin, ymin])
    maxs = np.array([xmax, ymax])

    X_range = maxs - mins

    X = X * X_range + mins
    return X


def new_save_folder(save_dir):
    # Specify the directory path to search for folders

    # Get a list of all folders in the specified directory
    folders = [f for f in os.listdir(save_dir) if os.path.isdir(os.path.join(save_dir, f))]

    # Filter out folders with non-integer names
    integer_folders = [int(folder) for folder in folders if folder.isdigit()]

    if len(integer_folders) == 0:
        # No folders present, create folder "1"
        new_folder_path = os.path.join(save_dir, '0')
    else:
        # Sort the integer folder names in ascending order
        integer_folders.sort()
        # Find the last folder name in the sorted list
        last_folder = integer_folders[-1]

        # Get the next folder name by incrementing the last folder name
        next_folder = str(int(last_folder) + 1)

        # Create the new folder path
        new_folder_path = os.path.join(save_dir, next_folder)

    # Create the new folder
    os.makedirs(new_folder_path)

    # Copy config file to save folder
    shutil.copy2("./config.py", new_folder_path)
    shutil.copy2("./utils.py", new_folder_path)
    cfg = Config()
    with open(f'{new_folder_path}/cfg.pkl', "wb") as f:
        pickle.dump(cfg, f)

    return new_folder_path


if __name__ == "__main__":
    Xs, _, _ = make_grid(19, (-2, 2, -2, 2))

    p = [bin_pd(X) for X in Xs]
    p = np.array(p).reshape(19, 19)

    plt.imshow(p, origin="lower", extent=(-2, 2, -2, 2))
    plt.yticks(np.linspace(-2, 2, 5))
    plt.xticks(np.linspace(-2, 2, 5))
    plt.tight_layout()
    plt.show()
