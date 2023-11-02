import math
import torch
import gpytorch as gpt
from matplotlib import pyplot as plt
from sklearn.datasets import make_moons
import numpy as np

num_tasks = 2


def moons_dataset():
    x, y = make_moons(n_samples=50, noise=0.1)
    x, y = torch.from_numpy(x).float(), torch.from_numpy(y)

    # x = torch.linspace(0, 1, 100)
    # y = torch.stack([
    #     torch.sin(x * (2 * math.pi)) + torch.randn(x.size()) * 0.2,
    #     torch.cos(x * (2 * math.pi)) + torch.randn(x.size()) * 0.2,
    #     torch.sin(x * (2 * math.pi)) + 2 * torch.cos(x * (2 * math.pi)) + torch.randn(x.size()) * 0.2,
    #     -torch.cos(x * (2 * math.pi)) + torch.randn(x.size()) * 0.2,
    # ], -1)

    return x, y


class MultitaskGPModel(gpt.models.ApproximateGP):
    def __init__(self, X_train):
        # Let's use a different set of inducing points for each latent function

        # We have to mark the CholeskyVariationalDistribution as batch
        # so that we learn a variational distribution for each task
        variational_distribution = gpt.variational.CholeskyVariationalDistribution(X_train.shape[0],
                                                                                   batch_shape=torch.Size([num_tasks]))

        # We have to wrap the VariationalStrategy in a LMCVariationalStrategy
        # so that the output will be a MultitaskMultivariateNormal rather than a batch output
        variational_strategy = gpt.variational.IndependentMultitaskVariationalStrategy(
            gpt.variational.UnwhitenedVariationalStrategy(
                self, X_train, variational_distribution, learn_inducing_locations=False),
            num_tasks=num_tasks
        )

        super().__init__(variational_strategy)

        # The mean and covariance modules should be batched so each task is separate
        self.mean_module = gpt.means.ZeroMean(batch_shape=torch.Size([num_tasks]))
        self.covar_module = gpt.kernels.ScaleKernel(
            gpt.kernels.RBFKernel(batch_shape=torch.Size([num_tasks])),
            batch_shape=torch.Size([num_tasks])
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpt.distributions.MultivariateNormal(mean_x, covar_x)


def test(model, likelihood, X_train, Y_train):
    model.eval(), likelihood.eval()
    # Test on grid of points
    h = 0.1
    xx1, xx2 = np.meshgrid(np.arange(-2, 2.5, h), np.arange(-2, 2, h))

    with torch.no_grad():
        X_test = torch.from_numpy(np.vstack((xx1.reshape(-1), xx2.reshape(-1))).T).float()
        f_pred = model(X_test)

        print(f_pred.mean.shape)

        mean_prob = torch.nn.functional.softmax(f_pred.mean, dim=-1)[:, 0]

        fg, ax = plt.subplots(1, 1, figsize=(4, 3))

        cs = ax.contourf(xx1, xx2, mean_prob.numpy().reshape(xx1.shape), levels=16)
        ax.scatter(X_train[Y_train == 0, 0], X_train[Y_train == 0, 1])
        ax.scatter(X_train[Y_train == 1, 0], X_train[Y_train == 1, 1])
        plt.colorbar(cs, ax=ax)
        plt.show()

    # Get fantasy model
    fant_x, fant_y = np.array([[0.1, 0.1]]), np.array([0])
    fantasy_model = model.get_fantasy_model(fant_x, fant_y)


def main():
    X_train, Y_train = moons_dataset()

    # Initialize model and likelihood
    model = MultitaskGPModel(X_train)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    likelihood = gpt.likelihoods.SoftmaxLikelihood(num_classes=2, num_features=1, mixing_weights=None)

    model.train(), likelihood.eval()

    for n, p  in model.named_parameters():
        if "raw_outputscale" in n:
            print(n)
            print(p.data.squeeze())
            p.data = 3 * torch.ones_like(p.data)
            print(p.data.squeeze())
            print()
        if "raw_lengthscale" in n:
            print(n)
            print(p.data.squeeze())
            p.data = -1 * torch.ones_like(p.data)
            print(p.data.squeeze())
            print()
    print()
    print()

    mll = gpt.mlls.VariationalELBO(likelihood, model, num_data=Y_train.size(0))

    for i in range(5000):
        optimizer.zero_grad()
        output = model(X_train)
        loss = -mll(output, Y_train)
        loss.backward()
        optimizer.step()

    for n, p  in model.named_parameters():
        if "covar_module" in n:
            print(n, p.data.squeeze())
    print()
    print()

    test(model, likelihood, X_train, Y_train)
    # model.eval(), likelihood.eval()
    # # Test on grid of points
    # h = 0.1
    # xx1, xx2 = np.meshgrid(np.arange(-2, 2.5, h), np.arange(-2, 2, h))
    #
    # with torch.no_grad():
    #     X_test = torch.from_numpy(np.vstack((xx1.reshape(-1), xx2.reshape(-1))).T).float()
    #     f_pred = model(X_test)
    #
    #     print(f_pred.variance.shape)
    #
    #     mean_prob = torch.nn.functional.softmax(f_pred.mean, dim=-1)[:, 0]
    #
    #     fg, ax = plt.subplots(1, 1, figsize=(4, 3))
    #
    #     cs = ax.contourf(xx1, xx2, mean_prob.numpy().reshape(xx1.shape), levels=16)
    #     ax.scatter(X_train[Y_train == 0, 0], X_train[Y_train == 0, 1])
    #     ax.scatter(X_train[Y_train == 1, 0], X_train[Y_train == 1, 1])
    #     plt.colorbar(cs, ax=ax)
    #     plt.show()
    # preds = likelihood(output)
    # print(f'{preds.probs.shape=}')
    # exit(56)


if __name__ == "__main__":
    np.random.seed(0)
    main()
