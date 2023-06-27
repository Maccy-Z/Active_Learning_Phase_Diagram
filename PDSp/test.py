import matplotlib
import GPy
import GPyOpt
import numpy
from AcquisitionNew import AcquisitionNew
from numpy.random import seed
from matplotlib import pyplot as plt
import numpy as np
from pretty_plot import plot_predcitions

def fun(x):
    x1 = x[:,0]
    x2 = x[:,1]
    return numpy.sign(x1**2 + x2**2 - 1)
    # if np.abs(x1) < 1:
    #     y = 1
    # else:
    #     y = -1
    # return y


def _create_model(self, X, Y):
    """
    Creates the model given some input data X and Y.
    """
    print(X, Y)
    # --- define kernel
    self.input_dim = X.shape[1]
    if self.kernel is None:
        kern = GPy.kern.Matern52(self.input_dim, variance=1., ARD=self.ARD) #+ GPy.kern.Bias(self.input_dim)
    else:
        kern = self.kernel
        self.kernel = None

    # --- define model
    noise_var = Y.var()*0.01 if self.noise_var is None else self.noise_var

    if not self.sparse:
        self.model = GPy.models.GPRegression(X, Y, kernel=kern, noise_var=noise_var, mean_function=self.mean_function)
    else:
        self.model = GPy.models.SparseGPRegression(X, Y, kernel=kern, num_inducing=self.num_inducing, mean_function=self.mean_function)

    # --- restrict variance if exact evaluations of the objective
    if self.exact_feval:
        self.model.Gaussian_noise.constrain_fixed(1e-6, warning=False)
    else:
        # --- We make sure we do not get ridiculously small residual noise variance
        self.model.Gaussian_noise.constrain_bounded(1e-9, 1e6, warning=False) #constrain_positive(warning=False)

objective = GPyOpt.core.task.SingleObjective(fun)

space = GPyOpt.Design_space(space=[{'name': 'x', 'type': 'continuous', 'domain': (-2,2)},
                                   {'name': 'y', 'type': 'continuous', 'domain': (-2,2)}])

model = GPyOpt.models.GPModel(optimize_restarts=5, verbose=False, kernel=GPy.kern.RBF(input_dim=2))
#model._create_model = lambda arg1, arg2: _create_model(model, arg1, arg2)

acquisition_optimizer = GPyOpt.optimization.AcquisitionOptimizer(space)

initial_design = GPyOpt.experiment_design.initial_design('random', space, 5)

acquisition = AcquisitionNew(model, space, optimizer=acquisition_optimizer)

evaluator = GPyOpt.core.evaluators.Sequential(acquisition)

bo = GPyOpt.methods.ModularBayesianOptimization(model, space, objective, acquisition, evaluator, initial_design, normalize_Y=False)

max_iter = 50
bo.run_optimization(max_iter = max_iter)

bo.plot_acquisition()

plot_predcitions(model)