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

    sample_old: int = None  # Samples for P_{n}
    sample_new: int = None  # Samples for P_{n+1}

    N_phases: int = 3

if __name__ == "__main__":
    print(Config())



