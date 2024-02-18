from dataclasses import dataclass
import math


@dataclass
class Config:
    """ Experiment setup """
    N_dim: int = 3  # Dimension of parameter space
    N_phases: int = 2
    extent: tuple = None  # Extent of parameter space to search. Set below.

    """Search resolution"""
    N_dist: int = 7  # Points distance function is evaluated at
    N_eval: int = 7  # Candidate points for new sample
    N_display: int = 11  # Number of points to visualise

    """Acquisition function parameters"""
    skip_point: float = 0.95  # Min prob to be confident enough to skip a point
    skip_phase: float = 0.05  # Min prob to skip sampling a phase
    sample_dist: float = 0.5  # Size of region to sample P_{n+1} over.

    """GP parameters"""
    obs_prob: float = 0.9  # Default certainty of observations. (Not too close to 1 or 0)
    gaus_herm_n: int = 10  # Number of samples for Gauss Hermite quadrature
    N_optim_steps: int = 250  # Number of steps to optimise hyperparameters

    """General Parameters"""
    N_CPUs: int = 8  # Number of CPUs to use

    def __post_init__(self):
        self.extent = ((-2, 2),
                       (-2, 2),
                       (-2, 2)
                       )
        self.unit_extent = tuple(((0, 1) for _ in range(self.N_dim)))

        # Assertions to make sure parameters are valid
        assert self.skip_phase <= (1 - self.skip_point), "skip_phase must be less than 1 - skip_point to ensure more than 1 phase is sampled"
        assert 0 < self.obs_prob < 1, "obs_P must be strictly between 0 and 1"
        assert len(self.extent) == self.N_dim, "Extent must have N_dim dimensions"


if __name__ == "__main__":
    print(Config())
