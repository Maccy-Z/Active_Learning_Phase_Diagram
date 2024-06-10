""" Plot observations from a save file"""
from src.utils import ObsHolder
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


obs_holder = ObsHolder.load("./saves/5")

obs_pos, obs_phse, obs_probs = obs_holder.get_og_obs()


print("Observation Locations")
print(obs_pos.tolist())
print("Observation Phases")
print(obs_phse.tolist())
print("Observation Probabilities")
print(obs_probs.tolist())

if obs_holder.cfg.N_dim == 3:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(obs_pos[:, 0], obs_pos[:, 1], obs_pos[:, 2], c=obs_phse, cmap='viridis')
    plt.show()
else:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(obs_pos[:, 0], obs_pos[:, 1], c=obs_phse, cmap='viridis')
    plt.show()



