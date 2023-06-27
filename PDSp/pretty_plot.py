from matplotlib import pyplot as plt
import numpy as np


def test_fn_boundary(theta):
    xs = np.cos(theta)
    ys = np.sin(theta)

    return np.array([xs, ys])


def plot_predcitions(model):
    """Takes in a model and plots training data and preditions"""

    n_point = 200
    X1 = np.linspace(-2, 2, n_point)
    X2 = np.linspace(-2, 2, n_point)
    x1, x2 = np.meshgrid(X1, X2)
    X = np.hstack((x1.reshape(n_point**2,1),x2.reshape(n_point**2,1)))

    m, v = model.predict(X)     # m.shape = [n**2, 1]. Retuns standard deviations?!

    plt.figure(figsize=(15,5))

    plt.subplot(1, 3, 1)
    plt.title('Posterior mean')

    # Plot predictions
    plt.contourf(X1, X2, m.reshape(200, 200), 100, vmin=-1.5, vmax=1.5)
    plt.colorbar(ticks=[-1, 0, 1])

    # Plot train points. Extract training data points from model.
    sample_X = model.model.X
    sample_Y = model.model.Y
    cmap = plt.cm.plasma
    plt.scatter(sample_X[:, 0], sample_X[:, 1], c=sample_Y, cmap=cmap)

    # Plot true boundary
    theta_values = np.linspace(0, 2 * np.pi, 100)
    points = test_fn_boundary(theta_values)
    plt.plot(points[0], points[1], label="True boundary")
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.title('STD from 0')

    std_from_0 = m / v
    # Plot predictions
    plt.contourf(X1, X2, std_from_0.reshape(200, 200), 100, vmin=-1, vmax=1)
    plt.colorbar(ticks=[-1, 0, 1])
    # Plot true boundary
    theta_values = np.linspace(0, 2 * np.pi, 100)
    points = test_fn_boundary(theta_values)
    plt.plot(points[0], points[1], label="True boundary")
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.title('Contour of mean=0')

    means = m.reshape(200, 200)
    plt.contour(X1, X2, means, levels=[0], colors='black')

    theta_values = np.linspace(0, 2 * np.pi, 100)
    points = test_fn_boundary(theta_values)
    plt.plot(points[0], points[1], label="True boundary")
    plt.legend()
    plt.grid(True)

    plt.show()






