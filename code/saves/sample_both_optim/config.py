from dataclasses import dataclass

@dataclass
class Config:
    xmin: int = -2
    xmax: int = 2
    ymin: int = -2
    ymax: int = 2

    N_init: int = 4
    N_dist: int = 19  # Points distance function is evaluated at
    N_eval: int = 15  # Candidate points for new sample
    N_display: int = 25 # Number of points to visualise

    sample_old: int = 100  # Samples for P_{n}
    sample_new: int = 100  # Samples for P_{n+1}
    skip_point: float = 0.95  # Min prob to entirely skip a point
    skip_phase: float = 0.01 # Min prob to skip sampling a phase
    sample_dist: float = None  # Size of region to sample P_{n+1} over.

    N_phases: int = 3

    optim_step: bool = True #  Optimise MLE when sampling x_{n+1}

    normal_sample: str = 'cholesky' # Modify generation of random normals for speed up

    noise_var: float = 0.001 # Variance for observations




if __name__ == "__main__":
    print(Config())



