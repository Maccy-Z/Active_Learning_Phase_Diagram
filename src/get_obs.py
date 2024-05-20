from src.utils import ObsHolder
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


obs_holder = ObsHolder.load("./saves/3")

obs_pos, obs_phse, obs_probs = obs_holder.get_og_obs()


print(obs_pos.tolist())
print()
print(obs_phse.tolist())
print()
print(obs_probs.tolist())

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(obs_pos[:, 0], obs_pos[:, 1], obs_pos[:, 2], c=obs_phse, cmap='viridis')
plt.show()



