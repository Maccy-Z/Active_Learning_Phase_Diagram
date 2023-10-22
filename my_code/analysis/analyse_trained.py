# Analyse trained GP models. Plot predictions at given time
import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
cwd = os.getcwd()
os.chdir(os.path.dirname(cwd))

from matplotlib import pyplot as plt
import numpy as np

from my_code.Distance_Sampler import DistanceSampler
from my_code.utils import ObsHolder

T = 100
obs_holder = ObsHolder.load("./saves/14")
obs_holder._obs_pos = obs_holder._obs_pos[:T]
obs_holder.obs_phase = obs_holder.obs_phase[:T]
cfg = obs_holder.cfg

distance_sampler = DistanceSampler(init_phases=obs_holder.obs_phase, init_Xs=obs_holder._obs_pos, cfg=cfg, save_dir="/tmp")

fig = plt.figure(figsize=(11, 3.3))

# Create three subplots
axes1 = fig.add_subplot(131)
axes2 = fig.add_subplot(132)
axes3 = fig.add_subplot(133)

distance_sampler.set_plots([axes1, axes2, axes3], fig)
distance_sampler.single_obs()

def set_ticks(ax):
    ax.set_xticks(np.linspace(-2, 2, 3), labels=np.linspace(-2, 2, 3).astype(int), fontsize=11)
    ax.set_yticks(np.linspace(-2, 2, 3), labels=np.linspace(-2, 2, 3).astype(int), fontsize=11)

set_ticks(axes1)
set_ticks(axes2)
set_ticks(axes3)

plt.subplots_adjust(left=0.04, right=0.99, top=0.92, bottom=0.08, wspace=0.15, hspace=0.4)

#fig.tight_layout()
fig.show()
