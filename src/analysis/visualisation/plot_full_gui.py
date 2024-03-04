import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(parent_dir)

cwd = os.getcwd()
os.chdir(os.path.dirname(cwd))
print(os.getcwd())

from src.DistanceSampler2D import DistanceSampler2D
from src.utils import quad_pd, ObsHolder
from matplotlib import pyplot as plt
import time
from dataclasses import dataclass
from copy import deepcopy

# Preset config file for this test
@dataclass
class Config:
    """ Experiment setup """
    N_dim: int = 2  # Dimension of parameter space
    N_phases: int = 4  # Number of phases to sample. Note: This is not checked in the code if set incorrectly.
    extent: tuple = None  # Extent of parameter space to search. Set below.

    """Search resolution"""
    N_dist: int = 31  # Points distance function is evaluated at
    N_eval: int = 31  # Candidate points for new sample
    N_display: int = 31  # Number of points to visualise

    """Acquisition function parameters"""
    skip_point: float = 0.90  # Min prob to be confident enough to skip a point
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
og_obs = ObsHolder.load(f'../saves/0')

Xs, Ys, prob = og_obs.get_og_obs()

for T in [102]:
    obs_pos = Xs[:T]
    obs_phase = Ys[:T]
    obs_prob = prob[:T]

    print(f'{obs_pos.shape = }, {obs_phase.shape = }, {obs_prob.shape = }')
    distance_sampler = DistanceSampler2D(obs_phase, obs_pos, cfg, init_probs=obs_prob, save_dir="/tmp")

    # Create three subplots
    fig = plt.figure(figsize=(9, 3))
    axes1 = fig.add_subplot(131)
    axes2 = fig.add_subplot(132)
    axes3 = fig.add_subplot(133)

    distance_sampler.set_plots([axes1, axes2, axes3], fig)
    distance_sampler.single_obs()

    fig.tight_layout()
    fig.savefig(f"../pics/full_{T}.pdf", dpi=300)
    fig.show()
