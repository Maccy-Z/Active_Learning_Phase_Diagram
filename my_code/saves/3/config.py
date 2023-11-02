from dataclasses import dataclass


@dataclass
class Config:
    N_dim: int = 3  # Dimension of parameter space
    N_phases: int = 3

    extent: tuple = None  # Extent of parameter space to search. Set below.

    N_dist: int = 11  # Points distance function is evaluated at
    N_eval: int = 11  # Candidate points for new sample
    N_display: int = 21  # Number of points to visualise

    sample_likelihood: bool = False  # Sample from likelihood instead of posterior
    sample_old: int = None  # Samples for P_{n}
    sample_new: int = None  # Samples for P_{n+1}
    skip_point: float = 0.91  # Min prob to entirely skip a point
    skip_phase: float = 0.05  # Min prob to skip sampling a phase
    sample_dist: float = 0.5  # Size of region to sample P_{n+1} over.

    optim_step: bool = True  # Optimise MLE when sampling x_{n+1}

    normal_sample: str = 'cholesky'  # Modify generation of random normals for speed up

    # noise_var: float = 5e-2  # Variance for observations2
    kern_var: float = 10
    kern_r: float = 0.5

    def __post_init__(self):
        self.extent = ((0, 1),
                       (0, 1),
                       (0, 1)
                        )
        self.unit_extent = tuple(((0, 1) for _ in range(self.N_dim)))


if __name__ == "__main__":
    print(Config())
