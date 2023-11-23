from dataclasses import dataclass


@dataclass
class Config:
    """ Experiment setup """
    N_dim: int = 2  # Dimension of parameter space
    N_phases: int = 3
    extent: tuple = None  # Extent of parameter space to search. Set below.

    """Search resolution"""
    N_dist: int = 21  # Points distance function is evaluated at
    N_eval: int = 21  # Candidate points for new sample
    N_display: int = 21  # Number of points to visualise

    """Acquisition function parameters"""
    sample_likelihood: bool = False  # Sample from likelihood instead of posterior
    sample_old: int = None  # Samples for P_{n}
    sample_new: int = None  # Samples for P_{n+1}
    skip_point: float = 0.91  # Min prob to entirely skip a point
    skip_phase: float = 0.15  # Min prob to skip sampling a phase
    sample_dist: float = 0.5  # Size of region to sample P_{n+1} over.

    """GP parameters"""
    gaus_herm_n: int = 12  # Number of samples for Gauss Hermite quadrature
    T: float = 1.  # Temperature for softmax
    N_optim_steps: int = 250  # Number of steps to optimise hyperparameters

    def __post_init__(self):
        self.extent = ((-2, 2),
                       (-2, 2),
                       #(0, 1)
                       )
        self.unit_extent = tuple(((0, 1) for _ in range(self.N_dim)))


if __name__ == "__main__":
    print(Config())
