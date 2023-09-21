from dataclasses import dataclass

@dataclass
class Config:
    # Params for baseline tests
    steps: int = 101
    N_init: int = 6

    xmin: int = -2
    xmax: int = 2
    ymin: int = -2
    ymax: int = 2

    N_dist: int = 19  # Points distance function is evaluated at
    N_eval: int = 19  # Candidate points for new sample
    N_display: int = 38  # Number of points to visualise

    sample_old: int = None  # Samples for P_{n}
    sample_new: int = None  # Samples for P_{n+1}
    skip_point: float = 0.9  # Min prob to entirely skip a point
    skip_phase: float = 0.05  # Min prob to skip sampling a phase
    sample_dist: float = 1  # Size of region to sample P_{n+1} over.

    N_phases: int = 4

    optim_step: bool = True  # Optimise MLE when sampling x_{n+1}

    normal_sample: str = 'cholesky'  # Modify generation of random normals for speed up

    # noise_var: float = 5e-2  # Variance for observations2
    kern_var: float = 10
    kern_r: float = 0.5

    def __post_init__(self):
        self.extent = (self.xmin, self.xmax, self.ymin, self.ymax)


if __name__ == "__main__":
    print(Config())
