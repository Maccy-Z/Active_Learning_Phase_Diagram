# Helpful functions
from matplotlib import pyplot as plt
import numpy as np
from config import Config
import scipy

class ObsHolder:
    def __init__(self, cfg:Config):
        self.cfg = cfg
        self.obs_pos = []
        self.obs_phase = []

    def make_obs(self, X):
        print(X, self.tri_pd(X))

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

    def plot_samples(self):
        Xs = np.array(self.obs_pos)
        obs = np.array(self.obs_phase)
        cmap = plt.cm.plasma

        plt.scatter(Xs[:, 0], Xs[:,1], s=20)
        # plt.xlim(self.cfg.xmin, self.cfg.xmax)
        # plt.ylim(self.cfg.ymin, self.cfg.ymax)



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


def plot_predcitions(model):
    """Takes in a model and plots training data and preditions"""

    n_point = 30
    X, X1, X2 = make_grid(n_point)
    m, v = model.predict(X)     # m.shape = [n**2, 1]. Retuns standard deviations?!

    theta_values = np.linspace(0, 2 * np.pi, 100)
    points = test_fn_boundary(theta_values)

    plt.figure(figsize=(15,5))

    plt.subplot(1, 3, 1)
    plt.title('Posterior mean')

    # Plot predictions
    plt.contourf(X1, X2, m.reshape(n_point, n_point), 100, vmin=-1.5, vmax=1.5)
    plt.colorbar(ticks=[-1, 0, 1])
    # Plot train points. Extract training data points from model.
    sample_X = model.X
    sample_Y = model.Y
    cmap = plt.cm.plasma
    plt.scatter(sample_X[:, 0], sample_X[:, 1], c=sample_Y, cmap=cmap)
    # Plot true boundary
    plt.plot(points[0], points[1], label="True boundary")
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.title('STD from 0')

    std_from_0 = m / v
    # Plot predictions
    plt.contourf(X1, X2, std_from_0.reshape(n_point, n_point), 100, vmin=-1, vmax=1)
    plt.colorbar(ticks=[-1, 0, 1])

    # Plot true boundary
    plt.plot(points[0], points[1], label="True boundary")
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.title('Contour of mean=0')

    means = m.reshape(n_point, n_point)
    plt.contour(X1, X2, means, levels=[0], colors='black')

    # Plot true boundary
    plt.plot(points[0], points[1], label="True boundary")
    plt.legend()
    plt.grid(True)

    plt.show()


def model_plot(model, sample=False):
    """Plot predictions for a single model"""

    n_point = 50
    X, X1, X2 = make_grid(n_point)

    if sample:
        y_pred = model.posterior_samples_f(X, full_cov=True, size=1)
    else:
        y_pred, _ = model.predict(X, full_cov=True)     # m.shape = [n**2, 1]. Retuns standard deviations?!

    print(y_pred.shape)
    plt.title(f'Posterior mean {sample = }')

    # Plot predictions
    plt.contourf(X1, X2, y_pred.reshape(n_point, n_point), 100, vmin=-1.5, vmax=1.5)
    plt.colorbar(ticks=[-1, 0, 1])
    # # Plot train points. Extract training data points from model.
    # sample_X = model.X
    # sample_Y = model.Y
    # cmap = plt.cm.plasma
    # plt.scatter(sample_X[:, 0], sample_X[:, 1], c=sample_Y, cmap=cmap, marker="x", s=10)

    plt.show()

# Rescale from physical coords to plot coords
def plot_scale(points, N, xmin, xmax, ymin, ymax):
    xs, ys = points[:, 0], points[:, 1]
    xs = xs * (N / (xmax - xmin)) + (N / 2) - 0.5
    ys = ys * (N / (ymax - ymin)) + (N / 2) - 0.5
    return  xs, ys

if __name__ == "__main__":
    sampler = ObsHolder(Config())

    grid, _, _ = make_grid(25)

    for xs in grid:
        sampler.make_obs(xs)
    sampler.plot_samples()



