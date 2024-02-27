import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

cwd = os.getcwd()
os.chdir(os.path.dirname(cwd))
print(os.getcwd())

from src.DistanceSampler2D import DistanceSampler2D
from src.utils import quad_pd
from matplotlib import pyplot as plt
import time
from dataclasses import dataclass


# Preset config file for this test
@dataclass
class Config:
    """ Experiment setup """
    N_dim: int = 2  # Dimension of parameter space
    N_phases: int = 4  # Number of phases to sample. Note: This is not checked in the code if set incorrectly.
    extent: tuple = None  # Extent of parameter space to search. Set below.

    """Search resolution"""
    N_dist: int = 19  # Points distance function is evaluated at
    N_eval: int = 19  # Candidate points for new sample
    N_display: int = 19  # Number of points to visualise

    """Acquisition function parameters"""
    skip_point: float = 0.95  # Min prob to be confident enough to skip a point
    skip_phase: float = 0.05  # Min prob to skip sampling a phase
    sample_dist: float = 0.5  # Size of region to sample P_{n+1} over.

    """GP parameters"""
    obs_prob: float = 0.95  # Default certainty of observations. (Not too close to 1 or 0)
    gaus_herm_n: int = 10  # Number of samples for Gauss Hermite quadrature
    N_optim_steps: int = 200  # Number of steps to optimise hyperparameters

    """General Parameters"""
    N_CPUs: int = 8  # Number of parallel CPU threads to use

    def __post_init__(self):
        self.extent = ((-2, 2),
                       (-2, 2),
                       )
        self.unit_extent = tuple(((0, 1) for _ in range(self.N_dim)))

        # Assertions to make sure parameters are valid
        assert 0 < self.obs_prob < 1, "obs_P must be strictly between 0 and 1"
        assert len(self.extent) == self.N_dim, "Extent must have N_dim dimensions"


pd_fn = quad_pd

save_dir = "../saves"
print(save_dir)
cfg = Config()

# Init observations to start off
X_init = [[0.0, -1.25], [0, 1.]]
phase_init = [pd_fn(X)[0] for X in X_init]
prob_init = [pd_fn(X)[1] for X in X_init]
print(phase_init)

distance_sampler = DistanceSampler2D(phase_init, X_init, cfg, init_probs=prob_init, save_dir=save_dir)

# Create three subplots
fig = plt.figure(figsize=(11, 3.3))
axes1 = fig.add_subplot(131)
axes2 = fig.add_subplot(132)
axes3 = fig.add_subplot(133)

distance_sampler.set_plots([axes1, axes2, axes3], fig)

for i in range(101):
    print()
    print("Step:", i)
    st = time.time()
    new_point, prob = distance_sampler.single_obs()
    print(f'Time taken: {time.time() - st:.2f}s')
    print(f'{new_point =}, {prob = }')
    obs_phase, _ = pd_fn(new_point, train=True)

    distance_sampler.add_obs(obs_phase, new_point, prob=0.99)

    # print(distance_sampler.obs_holder.get_og_obs())
    if i % 2 == 0:
        fig.show()
