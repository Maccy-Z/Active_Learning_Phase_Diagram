import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
cwd = os.getcwd()
os.chdir(os.path.dirname(cwd))

import numpy as np
from src.utils import ObsHolder


obs_holder = ObsHolder.load("../saves/12")

obs_pos, obs_phse, obs_probs = obs_holder.get_og_obs()


print(obs_pos.tolist())
print()
print(obs_phse.tolist())
print()
print(obs_probs.tolist())

