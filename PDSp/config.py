from dataclasses import dataclass

@dataclass
class Config:
    xmin: int = -2
    xmax: int = 2
    ymin: int = -2
    ymax: int = 2

    N_train: int = 4
    N_dist: int = 19  # Points distance function is evaluated at
    N_eval: int = 15  # Candidate points for new sample

    sample_old: int = None
    sample_new: int = None

    N_phases: int = 3

if __name__ == "__main__":
    print(Config())



