import torch
import math
from utils import c_print, ObsHolder
from config import Config
from matplotlib import pyplot as plt
import numpy as np
from likelihoods import gauss_hermite_quadrature


#
# class SimpleGPR(gpytorch.models.ExactGP):
#     def __init__(self, train_x, train_y, likelihood):
#         super(SimpleGPR, self).__init__(train_x, train_y, likelihood)
#         self.mean_module = gpytorch.means.ZeroMean()
#         self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
#
#     def forward(self, x):
#         mean_x = self.mean_module(x)
#         covar_x = self.covar_module(x)
#         return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
#
#
# class GPR:
#     model: SimpleGPR = None
#     train_x = None
#     train_y = None
#
#     def __init__(self, noise_var=None, kern_var=None, kern_r=None, mean=None):
#         noise = torch.ones(100) * 0.1
#         self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
#
#         # print(self.likelihood.noise_covar.FixedGaussianNoise)
#         if noise_var is not None:
#             self.likelihood.noise = noise_var
#             self.likelihood.noise_covar.raw_noise.requires_grad = False
#
#     def fit(self, train_x, train_y, n_iters=None):
#         self.train_x, self.train_y = train_x, train_y
#
#         self.model = SimpleGPR(train_x, train_y, self.likelihood)
#         self.model.train()
#
#         if n_iters is not None:
#             self.fit_params(n_iters)
#
#     def fit_params(self, n_iters):
#         print("Fitting Params")
#         mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
#         optimizer = torch.optim.Adam(self.model.parameters(), lr=0.1)
#
#         for i in range(n_iters):
#             optimizer.zero_grad()
#             output = self.model(self.train_x)
#             loss = -mll(output, self.train_y)
#             loss.backward()
#             optimizer.step()
#
#         print(f"Final Loss: {loss.item():.5g}")
#
#         # Print out final model parameters
#         noise = self.model.likelihood.noise_covar.noise
#         kern_r = self.model.covar_module.base_kernel.lengthscale
#         kern_scale = self.model.covar_module.outputscale
#
#         c_print(f"Noise: {noise.squeeze().detach():.3g}", color="green")
#         c_print(f"Kern R: {kern_r.squeeze().detach():.3g}", color="green")
#         c_print(f"Kern var: {kern_scale.squeeze().detach():.3g}", color="green")
#
#     def predict(self, test_x):
#         self.model.eval()
#         with torch.no_grad():
#             pred_dist = self.model(test_x)
#             pred_like = self.likelihood(pred_dist)
#
#         out = {"mean": pred_dist.mean, "var": pred_dist.variance, "conf_region": pred_like.confidence_region(), "cov_matrix": pred_dist.covariance_matrix}
#         return out
#
#     def predict_and_plot(self, test_x):
#         preds = self.predict(test_x)
#
#         test_mean = preds['mean']
#         lower, upper = preds['conf_region']
#
#         plt.scatter(self.train_x, self.train_y)
#         plt.plot(test_x, test_mean)
#         plt.fill_between(test_x.squeeze(), lower, upper, alpha=0.5)
#         plt.show()
#
#     def predict_mean(self, test_x):
#         preds = self.predict(test_x)
#         return preds['mean']


class ParamHolder(torch.nn.Module):
    """Holds parameters, with constraint that parameters are positive using a softplus transformation."""
    raw_kern_len: torch.nn.Parameter
    raw_kern_scale: torch.nn.Parameter
    raw_noise: torch.nn.Parameter

    def __init__(self, kern_len=math.log(2), kern_scale=math.log(2), noise_var=math.log(2)):
        super().__init__()

        self.raw_kern_len = torch.nn.Parameter(self.inv_softplus(torch.tensor(kern_len)))
        self.raw_kern_scale = torch.nn.Parameter(self.inv_softplus(torch.tensor(kern_scale)))

        if noise_var is None:
            self.raw_noise = torch.nn.Parameter(self.inv_softplus(torch.tensor(math.log(2))))
        else:
            self.raw_noise = torch.nn.Parameter(self.inv_softplus(torch.tensor(noise_var)), requires_grad=False)

    # Return parameters in their original scale
    def get_params(self):
        kern_len = torch.nn.functional.softplus(self.raw_kern_len) + 0.3
        kern_scale = torch.nn.functional.softplus(self.raw_kern_scale) + 1e-4
        noise = torch.nn.functional.softplus(self.raw_noise) + 1e-6
        return {'kern_len':kern_len, 'kern_scale': kern_scale, 'noise': noise}

    @staticmethod
    def inv_softplus(x: torch.Tensor):
        if x > 20:
            return x
        else:
            return torch.log(torch.exp(x) - 1)


def kern_rbf(param_holder, X1, X2):
    """
    X1.shape = (n, dim)
    X2.shape = (m, dim)
    """
    X1 = X1.unsqueeze(1)  # Shape: [n, 1, d]
    X2 = X2.unsqueeze(0)  # Shape: [1, m, d]

    params = param_holder.get_params()
    kern_scale, kern_len = params['kern_scale'], params['kern_len']

    sqdist = torch.norm(X1 - X2, dim=2, p=2).pow(2)
    k_r = kern_scale * torch.exp(-0.5 * sqdist / kern_len ** 2)

    return k_r


def kern_matern_12(param_holder, X1, X2):
    """
    X1.shape = (n, dim)
    X2.shape = (m, dim)
    """
    X1 = X1.unsqueeze(1)  # Shape: [n, 1, d]
    X2 = X2.unsqueeze(0)  # Shape: [1, m, d]

    params = param_holder.get_params()
    kern_scale, kern_len = params['kern_scale'], params['kern_len']

    # Calculate the pairwise Euclidean distance
    r = torch.norm(X1 - X2, dim=2, p=2)

    # Matérn 1/2 Kernel formula
    k_matern_half = kern_scale * torch.exp(-r / kern_len)

    return k_matern_half


def kern_matern_32(param_holder, X1, X2):
    """
    X1.shape = (n, dim)
    X2.shape = (m, dim)
    """
    X1 = X1.unsqueeze(1)  # Shape: [n, 1, d]
    X2 = X2.unsqueeze(0)  # Shape: [1, m, d]

    params = param_holder.get_params()
    kern_scale, kern_len = params['kern_scale'], params['kern_len']

    # Calculate the pairwise Euclidean distance
    r = torch.norm(X1 - X2, dim=2, p=2)

    # Matérn 3/2 Kernel formula
    sqrt_3_r_l = math.sqrt(3) * r / kern_len
    k_matern = kern_scale * (1 + sqrt_3_r_l) * torch.exp(-sqrt_3_r_l)

    return k_matern


def kern_matern_52(param_holder, X1, X2):
    """
    X1.shape = (n, dim)
    X2.shape = (m, dim)
    """
    X1 = X1.unsqueeze(1)  # Shape: [n, 1, d]
    X2 = X2.unsqueeze(0)  # Shape: [1, m, d]

    params = param_holder.get_params()
    kern_scale, kern_len = params['kern_scale'], params['kern_len']

    # Calculate the pairwise Euclidean distance
    r = torch.norm(X1 - X2, dim=2, p=2)

    # Matérn 5/2 Kernel formula
    sqrt_5_r_l = math.sqrt(5) * r / kern_len
    k_matern_five_half = kern_scale * (1 + sqrt_5_r_l + 5 * r.pow(2) / (3 * kern_len.pow(2))) * torch.exp(-sqrt_5_r_l)

    return k_matern_five_half


class GPRSimple(torch.nn.Module):
    def __init__(self, noise_var=None, kern_len=math.log(2), kern_scale=math.log(2)):
        super(GPRSimple, self).__init__()
        self.param_holder = ParamHolder(noise_var=noise_var, kern_len=kern_len, kern_scale=kern_scale)

        self.is_fitted = False

        self.kernel = lambda x, y: kern_matern_12(self.param_holder, x, y)

    def fit(self, train_x: torch.Tensor, train_y: torch.Tensor, n_iters=None):
        assert train_x.dim() == 2, "train_x must be a 2D tensor."
        self.train_x, self.train_y = train_x, train_y

        if n_iters is not None:
            self.fit_params(n_iters)

        noise = self.param_holder.get_params()['noise']

        # Cache important matrices
        K = self.kernel(train_x, train_x) + noise * torch.eye(len(train_x))
        self.L = torch.linalg.cholesky(K)
        self.inv_y = torch.cholesky_solve(train_y.unsqueeze(1), self.L)

        self.is_fitted = True

        return {param_name: param.detach().item() for param_name, param in self.param_holder.get_params().items()}

    # Log likelihood for fitting params
    def log_marginal_likelihood(self):
        noise = self.param_holder.get_params()['noise']
        K = self.kernel(self.train_x, self.train_x) + noise * torch.eye(len(self.train_x))
        L = torch.linalg.cholesky(K)
        inv_y = torch.cholesky_solve(self.train_y.unsqueeze(1), L)

        # Negative log-likelihood
        inv_quad = self.train_y.view(1, -1) @ inv_y
        logdet = torch.logdet(L)

        log_likelihood = 0.5 * (inv_quad + 2 * logdet + len(self.train_x) * torch.log(torch.tensor(2 * torch.pi)))

        # print(f'{inv_quad=}, {logdet=}')
        # print(torch.logdet(L))
        return log_likelihood / self.train_y.shape[0]

    def fit_params(self, n_iters):
        optim = torch.optim.Adam(self.parameters(), lr=0.1)
        for epoch in range(n_iters):
            optim.zero_grad()
            loss = self.log_marginal_likelihood()
            loss.backward()
            optim.step()

        params = self.param_holder.get_params()
        kern_len, kern_scale, noise = params['kern_len'], params['kern_scale'], params['noise']
        print()
        c_print(f'noise = {noise.item():.3g}', color="green")
        c_print(f'kern len = {kern_len.item():.3g}', color="green")
        c_print(f'kern var = {kern_scale.item():.3g}', color="green")

    def predict(self, pred_x: torch.Tensor):
        assert pred_x.dim() == 2, "pred_x must be a 2D tensor."
        if not self.is_fitted:
            raise RuntimeError("The model must be fitted before prediction.")

        noise = self.param_holder.get_params()['noise']

        # Compute the cross-covariance matrix K_*
        K_star = self.kernel(pred_x, self.train_x)

        # Predict mean
        pred_mean = K_star @ self.inv_y

        # Predict covariance
        v = torch.linalg.solve_triangular(self.L, K_star.T, upper=False)
        pred_cov = self.kernel(pred_x, pred_x) - v.T @ v

        # Diagoanl variance
        diag_var = torch.diag(pred_cov)
        # Confidence region including observation noise
        conf_region = (pred_mean.squeeze() - 2 * torch.sqrt(diag_var + noise), pred_mean.squeeze() + 2 * torch.sqrt(diag_var + noise))
        preds = {"mean": pred_mean.squeeze(), "var": diag_var, "cov_matrix": pred_cov, 'conf_region': conf_region}
        return preds

    def predict_and_plot(self, test_x):
        with torch.no_grad():
            preds = self.predict(test_x.view(-1, 1))

        test_mean = preds['mean']
        lower, upper = preds['conf_region']

        plt.scatter(self.train_x, self.train_y)
        plt.plot(test_x.squeeze(), test_mean)
        plt.fill_between(test_x.squeeze(), lower, upper, alpha=0.5)
        plt.show()


class GPRSoftmax(torch.nn.Module):
    models: list[GPRSimple]
    fitted_params: list[dict]   # Hyperparameters for each phase, [[kern_len, kern_scale, noise], ...]

    def __init__(self, obs_holder: ObsHolder, cfg: Config):
        super(GPRSoftmax, self).__init__()

        self.cfg = cfg
        self.obs_holder = obs_holder

        self.models = []
        self.fitted_params = []

    def make_GP(self, fit_hyperparam: bool = True):
        """
        For each phase, make a GPRSimple classifier
        """
        obs_X, obs_phase = self.obs_holder.get_obs()
        obs_X, obs_phase = torch.from_numpy(obs_X).to(torch.float), torch.from_numpy(obs_phase).to(torch.float)
        obs_X = obs_X.view(-1, self.cfg.N_dim)

        for phase in range(self.cfg.N_phases):
            obs_y = (obs_phase == phase).to(torch.float)
            logit_y = self._phase_to_mean(obs_y)
            phase_model = GPRSimple()

            if fit_hyperparam:
                params = phase_model.fit(obs_X, logit_y, n_iters=self.cfg.N_optim_steps)
                self.fitted_params.append(params)
            else:
                phase_model.fit(obs_X, logit_y, n_iters=None)

            self.models.append(phase_model)

        return self.fitted_params

    def predict(self, pred_x) -> np.ndarray:
        pred_x = torch.from_numpy(pred_x).to(torch.float)
        means, covs = [], []  # shape = [N_phases, N_pred]
        for model in self.models:
            pred = model.predict(pred_x)
            means.append(pred['mean'])
            covs.append(pred['var'])

        # self.models[-1].predict_and_plot(pred_x)

        means = torch.stack(means).T.detach().numpy()  # shape = [N_pred, N_phases]
        covs = torch.stack(covs).T.detach().numpy()
        # For each point, find prob of each phase using Gauss Hermite on likelihood
        probs = []
        for mu, cov in zip(means, covs):
            phase_probs = []
            for phase in range(self.cfg.N_phases - 1):  # Last phase can be calculated by 1 - sum(other phases)
                prob = gauss_hermite_quadrature(mu, cov, self.cfg.gaus_herm_n, T=self.cfg.T, phase_no=phase)
                phase_probs.append(prob)
            phase_probs.append(1 - sum(phase_probs))
            probs.append(phase_probs)
        probs = np.array(probs)

        # plt.plot(pred_x.squeeze(), probs[:, 0])
        # plt.show()

        # self.models[-1].predict_and_plot(pred_x)
        return probs

    def _phase_to_mean(self, bin_obs):
        logit_y = bin_obs * 2 - 1
        return logit_y.to(torch.float)

    def fantasy_GPs(self, fant_x: np.ndarray, fant_phase: np.ndarray):
        """
        Make new models assuming new observations fant_phase at fant_x
        fant_x.shape = [N_fant, N_dim]
        fant_phase.shape = [N_fant]

        Return new class with fantasy models
        """

        obs_X, obs_phase = self.obs_holder.get_obs()

        fant_x = fant_x.reshape(-1, self.cfg.N_dim)

        obs_X = np.concatenate([obs_X, fant_x])
        obs_phase = np.append(obs_phase, fant_phase)

        obs_X, obs_phase = torch.from_numpy(obs_X).to(torch.float), torch.from_numpy(obs_phase).to(torch.float)
        new_model = GPRFantasy(self.obs_holder, self.cfg, obs_X, obs_phase, self.fitted_params)

        return new_model


class GPRFantasy(GPRSoftmax):
    """
    Class for fantasy models.
    """

    def __init__(self, obs_holder: ObsHolder, cfg: Config, obs_X, obs_phase, fitted_params):
        """Make a fantasy model from existing observations and fit.
        Use parent hyperparameters"""
        super(GPRFantasy, self).__init__(obs_holder, cfg)

        for phase, params in enumerate(fitted_params):
            kern_len, kern_scale, noise = params['kern_len'], params['kern_scale'], params['noise']

            obs_y = (obs_phase == phase).to(torch.float)
            logit_y = self._phase_to_mean(obs_y)
            phase_model = GPRSimple(kern_len=kern_len, kern_scale=kern_scale, noise_var=noise)

            phase_model.fit(obs_X, logit_y, n_iters=None)

            self.models.append(phase_model)



def main():
    train_x = torch.linspace(0, 1, 25)
    train_y = (train_x > 0.8).to(int)  # torch.sin(train_x * (2 * math.pi)) + torch.randn(train_x.size()) * math.sqrt(0.001)
    test_x = torch.linspace(0, 2, 100).view(-1, 1)

    obs_holder = ObsHolder(Config(), "test")

    for x, y in zip(train_x, train_y):
        obs_holder.make_obs(x, y)

    model = GPRSoftmax(obs_holder, Config())
    hps = model.make_GP()
    print("Making Fantasy")
    new_model = model.fantasy_GPs(torch.tensor([[2.]]), torch.tensor([3]))
    print("Predicting")
    new_model.predict(test_x)


if __name__ == "__main__":
    main()
