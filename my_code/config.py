from dataclasses import dataclass
import math


@dataclass
class Config:
    """ Experiment setup """
    N_dim: int = 2  # Dimension of parameter space
    N_phases: int = 3
    extent: tuple = None  # Extent of parameter space to search. Set below.

    """Search resolution"""
    N_dist: int = 7  # Points distance function is evaluated at
    N_eval: int = 7  # Candidate points for new sample
    N_display: int = 11  # Number of points to visualise

    """Acquisition function parameters"""
    skip_point: float = 0.8  # Min prob to be confident enough to skip a point
    skip_phase: float = 0.19  # Min prob to skip sampling a phase
    sample_dist: float = 0.5  # Size of region to sample P_{n+1} over.

    """GP parameters"""
    obs_P: float = 0.9  # Certainty of observations
    gaus_herm_n: int = 10  # Number of samples for Gauss Hermite quadrature
    N_optim_steps: int = 250  # Number of steps to optimise hyperparameters

    """General Parameters"""
    N_CPUs: int = 8  # Number of CPUs to use

    def __post_init__(self):
        self.extent = ((-2, 2),
                       (-2, 2),
                       #(-2, 2)
                       )
        self.unit_extent = tuple(((0, 1) for _ in range(self.N_dim)))

        # We fit the GPs to be +- 1, and set the temp of the softmax to fix observation probability
        self.T = 1 / 2 * math.log((self.N_phases - 1) * self.obs_P / (1 - self.obs_P))

        # Assertions to make sure parameters are valid
        assert self.skip_phase <= (1 - self.skip_point), "skip_phase must be less than 1 - skip_point to ensure more than 1 phase is sampled"
        assert 0 < self.obs_P < 1, "obs_P must be strictly between 0 and 1"
        assert len(self.extent) == self.N_dim, "Extent must have N_dim dimensions"


if __name__ == "__main__":
    print(Config())
