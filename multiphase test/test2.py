import math
import torch
import random
import numpy as np
import gpytorch as gp

from matplotlib import pyplot as plt

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

from sklearn.datasets import make_moons

X, y = make_moons(noise = 0.3)
X_train = torch.from_numpy(X).float()
y_train = torch.from_numpy(y).float()

class GPClassificationModel(gp.models.ApproximateGP):
    def __init__(self, X_train):
        variational_distribution = gp.variational.CholeskyVariationalDistribution(
            X_train.size(0)
        )
        variational_strategy = gp.variational.VariationalStrategy(
            self, X_train, variational_distribution, learn_inducing_locations = True
        )

        super(GPClassificationModel, self).__init__(variational_strategy)

        self.mean_module = gp.means.ConstantMean()
        self.covar_module = gp.kernels.ScaleKernel(gp.kernels.RBFKernel())

    def forward(self, X):
        mean_X = self.mean_module(X)
        covar_X = self.covar_module(X)

        return gp.distributions.MultivariateNormal(mean_X, covar_X)

likelihood = gp.likelihoods.BernoulliLikelihood()
model = GPClassificationModel(X_train)

likelihood.train()
model.train()

train_iter = 50

optimizer = torch.optim.Adam(model.parameters(), lr = 0.1)

marginal_log_likelihood = gp.mlls.VariationalELBO(likelihood, model, y_train.numel())

for i in range(train_iter):
    optimizer.zero_grad()

    f_post = model(X_train)
    loss = - marginal_log_likelihood(f_post, y_train)
    loss.backward()

    print('Iter %d/%d - Loss: %.3f' % (i + 1, train_iter, loss.item()))
    optimizer.step()


likelihood.eval()
model.eval()

h = 0.05
x1_min, x1_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
x2_min, x2_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, h), np.arange(x2_min, x2_max, h))

with torch.no_grad():
    X_test = torch.from_numpy(np.vstack((xx1.reshape(-1), xx2.reshape(-1))).T).float()
    f_pred = model(X_test)
    y_pred = likelihood(f_pred)


with torch.no_grad():
    fg, ax = plt.subplots(1, 1, figsize = (4, 3))

    ax.contourf(xx1, xx2, y_pred.mean.numpy().reshape(xx1.shape), levels = 16)
    ax.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1])
    ax.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1])

    plt.show()