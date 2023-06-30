from matplotlib import pyplot as plt
import numpy as np
from collections import defaultdict

def fun(x):
    x1 = x[:,0]
    x2 = x[:,1]
    return np.sign(x1**2 + x2**2 - 1)


# Computes distance between two phase diagrams
def dist(pd_1: np.ndarray, pd_2: np.ndarray):
    # p1.shape = [n_points, 3]  p1[0] = [x1, x2, phase]
    n_points = pd_1.shape[0]
    p_1, p_2 = pd_1[:, 2].astype(int), pd_2[:, 2].astype(int)

    phase_1, area_1 = np.unique(p_1, return_counts=True)
    phase_2, area_2 = np.unique(p_2, return_counts=True)

    # Be careful if there is a missing phase in a diagram.
    p_area_1, p_area_2  = defaultdict(int), defaultdict(int)
    p_area_1.update(zip(phase_1, area_1))
    p_area_2.update(zip(phase_2, area_2))

    # Find all present phases in either diagram
    all_phases = np.union1d(phase_1, phase_2)

    abs_diffs = []
    for phase in all_phases:
        abs_diff = np.abs(p_area_1[phase] - p_area_2[phase])
        abs_diffs.append(abs_diff)

    tot_abs_diff = np.sum(abs_diffs) / n_points / 2

    return tot_abs_diff

def main():
    n_point = 100
    X1 = np.linspace(-1, 1, n_point)
    X2 = np.linspace(-1, 1, n_point)
    x1, x2 = np.meshgrid(X1, X2)
    xs = np.hstack((x1.reshape(n_point**2,1),x2.reshape(n_point**2,1)))

    p1 = np.expand_dims(fun(xs), axis=-1)
    p1 = np.concatenate([xs, p1], axis=1)

    p2 = p1
    print("Total distance:", dist(p1, p2))



if __name__ == "__main__":
    main()


