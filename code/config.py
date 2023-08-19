from dataclasses import dataclass


@dataclass
class Config:
    steps: int = 51

    xmin: int = -2
    xmax: int = 2
    ymin: int = -2
    ymax: int = 2

    N_init: int = 4
    N_dist: int = 25  # Points distance function is evaluated at
    N_eval: int = 19  # Candidate points for new sample
    N_display: int = 25 # Number of points to visualise

    sample_old: int = None  # Samples for P_{n}
    sample_new: int = None  # Samples for P_{n+1}
    skip_point: float = 0.7 # Min prob to entirely skip a point
    skip_phase: float = 0.05 # Min prob to skip sampling a phase
    sample_dist: float = 3  # Size of region to sample P_{n+1} over.

    N_phases: int = 3

    optim_step: bool = False #  Optimise MLE when sampling x_{n+1}

    normal_sample: str = 'cholesky' # Modify generation of random normals for speed up

    noise_var: float = 5e-2 # Variance for observations2
    kern_var: float = 4
    kern_r: float = 0.8

    def __post_init__(self):
        self.extent = (self.xmin, self.xmax, self.ymin, self.ymax)

if __name__ == "__main__":
    print(Config())



