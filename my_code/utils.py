# Helpful functions
from matplotlib import pyplot as plt
import numpy as np
import scipy
import os
import pickle
import shutil
import importlib.util
from matplotlib import ticker
from functools import lru_cache

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
        bounds = np.array(self.cfg.extent)
        mins, maxs = bounds[:, 0], bounds[:, 1]

        bbox_sizes = maxs - mins

        obs_pos = (obs_pos - mins) / bbox_sizes

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


# Make nxn...nx grid
@lru_cache(maxsize=10)
def make_grid(n, extent: tuple[tuple[float]]) -> (np.ndarray, np.ndarray):
    extent = np.array(extent)
    lin_spaces = [np.linspace(start, end, n) for start, end in extent]
    mesh = np.meshgrid(*lin_spaces, indexing='ij')

    # Reshape the meshgrid to have a list of points
    grid_shape = mesh[0].shape  # Shape of the n-dimensional grid
    num_points = np.prod(grid_shape)  # Total number of points

    # Create an array of points
    points = np.vstack([mesh_dim.reshape(num_points) for mesh_dim in mesh]).T

    return points, mesh


# Select points within a set distance from a give point
def points_within_radius(points: np.ndarray, target_point: np.ndarray, r_max: float, cube_limits: tuple[tuple[float]]):
    """
    Select points within a certain radius from a target point within an n-dimensional cube.

    Parameters:
    points (np.ndarray): Array of points in the n-dimensional cube, shape (num_points, n).
    target_point (np.ndarray or list): The reference point in n-dimensional space.
    r_max (float): The maximum distance from the target point.
    cube_limits (list): List of tuples defining the min and max values for each dimension.

    Returns:
    np.ndarray: Mask of wanted points.
    """

    # Step 1: Calculate the squared Euclidean distances
    # We use squared distances for comparison to avoid the computational cost of square roots
    squared_distances = np.sum((points - target_point) ** 2, axis=1)

    # Step 2: Filter points based on squared distance
    within_radius = squared_distances <= r_max ** 2

    # Check points within cube limits
    min_limits, max_limits = np.array(cube_limits).T  # Unpack limits
    within_cube = np.all((points >= min_limits) & (points <= max_limits), axis=1)

    # Combine the conditions
    mask = within_radius & within_cube

    return mask


# Rescale from physical coords to plot coords
def to_real_scale(X, extent):
    extent = np.array(extent)
    mins, maxs = extent[:, 0], extent[:, 1]

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


# Custom formatter for colorbar ticks
class CustomScalarFormatter(ticker.ScalarFormatter):
    def _set_format(self):  # Override function that finds format to use.
        self.format = "%1.1f"  # Give format here

if __name__ == "__main__":
    obs_holder: ObsHolder = ObsHolder.load("./saves/152")

    print(obs_holder.get_obs())
