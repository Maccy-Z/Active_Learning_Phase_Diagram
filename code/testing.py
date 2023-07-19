import edit_source_files
import pickle
from utils import ObsHolder
import GPy
import numpy as np
from utils import make_grid, Config
import math
from matplotlib import pyplot as plt

cfg = Config()

# Plot a grid of samples
def plot_samples(obs_holder, n, points=25, t=None):
    # Observations up to time t
    if t is not None:
        T = t + obs_holder.cfg.N_init ** 2
        obs_holder.obs_pos = obs_holder.obs_pos[:T]
        obs_holder.obs_phase = obs_holder.obs_phase[:T]

    Xs = np.array(obs_holder.obs_pos)
    obs = np.array(obs_holder.obs_phase)

    plot_Xs, X1, X2 = make_grid(points)

    models = fit_gp(obs_holder)
    pds, vars = single_pds(models, plot_Xs, sample=None)

    print(pds.shape, vars.shape)
    pds = np.concatenate([pds, vars], axis=0)


    n_img = len(pds)
    num_rows = math.isqrt(n_img)
    num_cols = math.ceil(n_img / num_rows)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(8, 8))
    for i, ax in enumerate(axes.flatten()):
        if i < n_img:
            pd = pds[i]
            c = ax.imshow(pd.reshape(points, points), extent=(-2, 2, -2, 2), origin="lower")  # Phase diagram
            ax.scatter(Xs[:, 0], Xs[:, 1], marker="x", s=40, c=obs, cmap='bwr')  # Existing observations
            ax.axis("off")

        else:
            print("Removing")
            ax.set_axis_off()

    plt.subplots_adjust(hspace=0.3)
    # plt.tight_layout()

    fig.colorbar(c)

    if t is not None:
        fig.suptitle(f"Samples of phase diagrams after {T} samples")
    else:
        fig.suptitle(f'Samples of phase diagrams')
    plt.show()


# Sample phase diagrams from models.
def single_pds(models: list[GPy.core.GP], xs, sample=None):
    """
    @param models: List of models for each phase
    @param xs: Points to sample from
    @param sample: Use mean or sample from GP.
    @return: List of possible phase diagrams.
    """
    y_preds, y_vars = [], []
    for phase_i, pd_model in enumerate(models):
        if sample is None:
            y_pred, var = pd_model.predict(xs, include_likelihood=False)  # m.shape = [n**2, 1]
        else:
            s = sample // 3
            y_pred = pd_model.posterior_samples_f(xs, size=s, method=cfg.normal_sample).squeeze(axis=1)
            var = 0.

        y_preds.append(y_pred)
        y_vars.append(var)

    y_preds = np.stack(y_preds)
    y_vars = np.stack(y_vars)

    # sample_pds = np.argmax(y_preds, axis=0).T
    sample_pds = y_preds.swapaxes(1, 2).reshape(-1, 625)
    # print(y_vars)
    # print(y_vars.shape)
    sample_vars = y_vars.swapaxes(1, 2).reshape(-1, 625)
    return sample_pds, sample_vars


# Sample phase diagrams from models.
def sample_pds(models: list[GPy.core.GP], xs, sample=None):
    """
    @param models: List of models for each phase
    @param xs: Points to sample from
    @param sample: Use mean or sample from GP.
    @return: List of possible phase diagrams.
    """
    y_preds = []
    for phase_i, pd_model in enumerate(models):
        if sample is None:
            y_pred, _ = pd_model.predict(xs)  # m.shape = [n**2, 1]
        else:
            s = sample // 3
            y_pred = pd_model.posterior_samples_f(xs, size=s, method=cfg.normal_sample).squeeze(axis=1)

        y_preds.append(y_pred)

    y_preds = np.stack(y_preds)

    # sample_pds = np.argmax(y_preds, axis=0).T
    sample_pds = y_preds.swapaxes(1, 2).reshape(-1, 625)
    return sample_pds


# Fit model to existing observations
def fit_gp(obs_holder, ) -> list[GPy.core.GP]:
    "Trains a model for each phase."
    X, Y = obs_holder.get_obs()

    models = []
    print()
    for i in range(3):
        phase_i = (Y == i) * 2 - 1  # Between -1 and 1

        kernel = GPy.kern.Matern52(input_dim=2, variance=1, lengthscale=1)
        model = GPy.models.GPClassification(X, phase_i.reshape(-1, 1), kernel)
        # model.optimize()

        # if kernel.lengthscale < 0.3:
        #     kernel.lengthscale = 0.3
        # if kernel.variance > 1:
        #     kernel.variance = 1.
        # print(kernel)

        models.append(model)
    return models



save_name = "17"

for t in range(0, 30, 5):
    print(t+16)
    print()
    with open(f'./saves/{save_name}/obs_holder', "rb") as f:
        obs_holder: ObsHolder = pickle.load(f)
        plot_samples(obs_holder, 9, t=t)
#
# import GPy
# import numpy as np
#
# kernel = GPy.kern.Matern52(input_dim=1, variance=1., lengthscale=100)
#
# x1 = np.array([[i] for i in range(5)])
# x2 = np.array([[0] for i in range(5)])
#
# K = kernel.K(x1, X2=x2)
# K = K.T[0]
# print(K)
#
#
