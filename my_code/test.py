from utils import ObsHolder

obs_holder = ObsHolder.load("./saves/0")

print(obs_holder.get_og_obs())

