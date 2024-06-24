# Evaluate error of GP model
import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from src.utils import make_grid, to_real_scale
from src.config import Config
from pd import skyrmion_pd_3D, skyrmion_pd_2D

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

Extent = ((0, 1.), (0., 1.), (0.1, 1.))
n_dim = 3
grid = (11, 11, 10)


class GridSearch:
    def __init__(self, full_dimensions=(11, 11, 10), low_res_dimensions=(3, 3, 4)):
        self.extent = np.array(Extent)

        self.full_dimensions = np.array(full_dimensions)
        self.low_res_dimensions = np.array(low_res_dimensions)
        self.low_res_total_points = np.prod(low_res_dimensions)
        self.full_total_points = np.prod(full_dimensions)
        self.low_res_grid = np.indices(low_res_dimensions).reshape(n_dim, -1).T
        self.full_grid = np.indices(full_dimensions).reshape(n_dim, -1).T
        self.current_index = 0
        self.low_res_done = False

    def __iter__(self):
        return self

    def __next__(self):
        if not self.low_res_done:
            if self.current_index < self.low_res_total_points:
                point = self.low_res_grid[self.current_index]
                self.current_index += 1
                return self.extent[:, 0] + (self.extent[:, 1] - self.extent[:, 0]) * point / (self.low_res_dimensions-1)
            else:
                self.low_res_done = True
                self.current_index = 0

        if self.current_index < self.full_total_points:
            point = self.full_grid[self.current_index]
            self.current_index += 1

            # print(f'{point = }')

            return self.extent[:, 0] + point * (1 / 10)
        else:
            raise StopIteration


def suggest_point(Xs, labels):
    # Train the SVC
    clf = KNeighborsClassifier(n_neighbors=1)
    clf.fit(Xs, labels)

    # Create a grid of points
    grid_points, mesh = make_grid(grid, Extent)

    # Get the decision function for each point on the grid
    Z = clf.predict(grid_points)
    y_pred = Z.reshape(*grid, -1)
    return y_pred


def plot(Xs, labels, new_point, contours, pred_pd):
    pred_pd = pred_pd[:, :, 5, 0].T
    mask = (Xs[:, 2] > 0.4) & (Xs[:, 2] < 0.6)
    Xs = Xs[mask]
    labels = np.array(labels)[mask]

    # Plot the results
    plt.figure(figsize=(3, 3))

    plt.scatter(Xs[:, 0], Xs[:, 1], marker="x", s=30, c=labels, cmap='bwr')  # , c=labels, cmap='viridis')
    plt.imshow(pred_pd, extent=(0, 1, 0, 1), origin="lower")  # Phase diagram
    if new_point is not None:
        plt.scatter(new_point[0], new_point[1], c='r')
    if contours is not None:
        for contor in contours:
            plt.plot(*zip(*contor), c='k', linestyle='dashed')

    plt.xlim([0, 1])
    plt.ylim([0, 1])
    # plt.yticks(np.linspace(-2, 2, 5))
    # plt.xticks(np.linspace(-2, 2, 5))
    plt.subplots_adjust(left=0.1, right=0.99, top=0.925, bottom=0.1, wspace=0.4, hspace=0.4)

    plt.show()


def error(pred_pd, pd_fn):
    points, _ = make_grid(grid, Extent)
    # points = to_real_scale(points, Extent)
    true_pd = []
    for X in points:
        true_phase = pd_fn(X)
        true_pd.append(true_phase)

    true_pd = np.stack(true_pd).reshape(pred_pd.shape)

    diff = np.not_equal(pred_pd, true_pd)
    diff_mean = np.mean(diff)
    #
    # print(points)
    # plot(np.array([[0, 0]]), np.array([0]), None, None, true_pd)
    return diff_mean


def main():
    pd_fn = skyrmion_pd_3D

    Xs = [[0.3, 0.4, 0.5, 0.6, 0, 0.7], [0, 0.1, 0, 0.5, 0, 0.8], [0.1, 0.1, 0.5, 0.3, 0.4, 1]]
    # Xs = [[0.05, 0.45, 0.1, 0.6, 0.2, 0.8, 1], [0.3, 0.05, 0.6, 0.3, 0.8, 0.5, 1]]
    Xs = np.array(Xs).T

    labels = [pd_fn(X) for X in Xs]
    print(labels)
    errors = []
    for i, new_point in enumerate(GridSearch()):
        # print(f'{new_point = }')
        full_pd = suggest_point(Xs, labels)
        Xs = np.append(Xs, [new_point], axis=0)
        labels.append(pd_fn(new_point))

        pred_error = error(full_pd, pd_fn)
        errors.append(pred_error)
        if i % 50 == 0:
            # plot(Xs, labels, new_point, None, full_pd)
            print(pred_error)

    # plt.imshow(full_pd, origin="lower")
    # plt.show()

    print(errors)
    print(len(errors))

    plot(Xs, labels, None, None, full_pd)


if __name__ == "__main__":
    main()
