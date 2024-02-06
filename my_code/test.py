from utils import ObsHolder

obs_holder = ObsHolder.load("./saves/11")

print(obs_holder.get_og_obs())

