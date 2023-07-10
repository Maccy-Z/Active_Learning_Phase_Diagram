from dataclasses import dataclass

@dataclass
class Config:
    xmin: int = -2
    xmax: int = 2
    ymin: int = -2
    ymax: int = 2

    N_train: int = 9
    N_dist: int = 25
    N_eval: int = 19

    sample_old: int = 200
    sample_new: int = 200

if __name__ == "__main__":
    print(Config())



