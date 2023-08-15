from dataclasses import dataclass
import numpy as np

@dataclass
class Config:
    steps: int = 31

    xmin: int = -3
    xmax: int = 3
    ymin: int = -2
    ymax: int = 2

    N_init: int = 4
    N_dist: int = 25  # Points distance function is evaluated at
    N_eval: int = 24  # Candidate points for new sample
    N_display: int = 25 # Number of points to visualise

    sample_old: int = None  # Samples for P_{n}
    sample_new: int = None  # Samples for P_{n+1}
    skip_point: float = 0.90 # Min prob to entirely skip a point
    skip_phase: float = 0.005 # Min prob to skip sampling a phase
    sample_dist: float = 3  # Size of region to sample P_{n+1} over.

    N_phases: int = 3

    optim_step: bool = False #  Optimise MLE when sampling x_{n+1}

    normal_sample: str = 'cholesky' # Modify generation of random normals for speed up

    noise_var: float = 5e-2 # Variance for observations2
    kern_var: float = 0.25
    kern_r: float = 0.9

    def get_kern_param(self, k_var, k_r, t=None):
        if t is None:
            step = len(self.obs_phase)
        else:
            step = t

        var = k_var + ((step - 16) * .75 / 50)
        r = k_r #  / (np.sqrt(step + 16)) + 0.25

        return var, r

    def __post_init__(self):
        self.extent = (self.xmin, self.xmax, self.ymin, self.ymax)

if __name__ == "__main__":
    print(Config())



