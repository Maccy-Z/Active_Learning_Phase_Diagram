# Helpful functions
from matplotlib import pyplot as plt
import numpy as np
from config import Config
import scipy
import os
import pickle

class ObsHolder:
    def __init__(self, cfg:Config):
        self.cfg = cfg
        self.obs_pos = []
        self.obs_phase = []

    def make_obs(self, X):
        self.obs_pos.append(X)
        self.obs_phase.append(self.tri_pd(X))

    def get_obs(self) ->[np.ndarray, np.ndarray]:

        return np.array(self.obs_pos), np.array(self.obs_phase)


    # Test function, representing making a observation
    def tri_pd(self, X):
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

    def plot_samples(self, show_obs=False):
        Xs = np.array(self.obs_pos)
        obs = np.array(self.obs_phase)

        if show_obs:
            plt.scatter(Xs[:, 0], Xs[:,1], s=20)

        # Interpolate points onto a grid
        xi = np.linspace(-2, 2, 100)  # x-coordinate of regular grid
        yi = np.linspace(-2, 2, 100)  # y-coordinate of regular grid
        xi, yi = np.meshgrid(xi, yi)  # Create grid mesh
        zi = scipy.interpolate.griddata(Xs, obs, (xi, yi), method='nearest')  # Interpolate using linear method

        plt.imshow(zi, origin="lower", extent=(-2, 2, -2, 2))
        #print(Xs[:, 0], Xs[:,1])
        #plt.tricontourf(Xs[:, 0], Xs[:,1], obs, levels=100)

        plt.xlim(self.cfg.xmin, self.cfg.xmax)
        plt.ylim(self.cfg.ymin, self.cfg.ymax)
        plt.show()

    def save(self, save_file):
        with open(f'{save_file}/obs_holder', "wb") as f:
            pickle.dump(self, f)



def make_grid(n, xmin=-2, xmax=2, ymin=-2, ymax=2) -> (np.ndarray, np.ndarray, np.ndarray):
    "very commonly used test to make a n x n grid or points"
    X1 = np.linspace(xmin, xmax, n)
    X2 = np.linspace(ymin, ymax, n)
    x1, x2 = np.meshgrid(X1, X2)
    X = np.hstack((x1.reshape(n ** 2, 1), x2.reshape(n ** 2, 1)))
    return X, X1, X2


def test_fn_boundary(theta):
    xs = np.cos(theta)
    ys = np.sin(theta)

    return np.array([xs, ys])



# Rescale from physical coords to plot coords
def plot_scale(points, N, xmin, xmax, ymin, ymax):
    xs, ys = points[:, 0], points[:, 1]
    xs = xs * (N / (xmax - xmin)) + (N / 2) - 0.5
    ys = ys * (N / (ymax - ymin)) + (N / 2) - 0.5
    return  xs, ys

def array_scale(X, N, min, max):
    n = (N-1) * (X - min) / (max - min)
    return n



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


    return new_folder_path

if __name__ == "__main__":
    sampler = ObsHolder(Config())

    grid, _, _ = make_grid(25)

    for xs in grid:
        sampler.make_obs(xs)
    sampler.plot_samples()

